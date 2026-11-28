"""
Direct tests of the synchronous ``run_*`` wrappers in
:mod:`rsgispy.dataaccess.planet_data_api`.

The ``planet`` package is replaced by a small fake (filters, order requests,
``Session`` and the ``data`` / ``orders`` clients) that records every call, so
the tests check what rsgispy sends to the Planet SDK and what it returns
without the SDK being installed and without any network access.
"""

import asyncio
import datetime
import os
import types

import pytest

import rsgispy
import rsgispy.dataaccess.planet_data_api as pl

AUTH = object()


class _Recorder:
    def __init__(self):
        self.calls = []
        self.sessions = []

    def add(self, _call_name, /, *args, **kwargs):
        self.calls.append((_call_name, args, kwargs))

    def names(self):
        return [c[0] for c in self.calls]

    def get(self, name):
        hits = [c for c in self.calls if c[0] == name]
        assert len(hits) == 1, f"{name} called {len(hits)} times"
        return hits[0]


def _make_fake_planet(rec, *, order_state="success", list_mode="aiter",
                      cancel_awaitable=True, n_items=3):
    """Return a fake ``planet`` module recording calls in ``rec``."""

    def _filter(kind):
        def _fn(*args, **kwargs):
            rec.add(f"filter.{kind}", *args, **kwargs)
            return (kind, args, tuple(sorted(kwargs.items())))
        return _fn

    data_filter = types.SimpleNamespace(
        permission_filter=_filter("permission"),
        geometry_filter=_filter("geometry"),
        date_range_filter=_filter("date_range"),
        range_filter=_filter("range"),
        and_filter=_filter("and"),
    )

    def _product(**kwargs):
        rec.add("order.product", **kwargs)
        return {"product": kwargs}

    def _notifications(**kwargs):
        rec.add("order.notifications", **kwargs)
        return {"notifications": kwargs}

    def _build_request(**kwargs):
        rec.add("order.build_request", **kwargs)
        return {"request": kwargs}

    order_request = types.SimpleNamespace(
        product=_product, notifications=_notifications, build_request=_build_request
    )

    class _DataClient:
        async def search(self, **kwargs):
            rec.add("data.search", **kwargs)
            for idx in range(min(n_items, kwargs["limit"])):
                yield {"id": f"item{idx}"}

    class _OrdersClient:
        async def create_order(self, request):
            rec.add("orders.create_order", request)
            return {"id": "order-1", "state": "queued", "request": request}

        async def get_order(self, order_id):
            rec.add("orders.get_order", order_id)
            return {"id": order_id, "state": order_state}

        async def download_order(self, order_id, **kwargs):
            rec.add("orders.download_order", order_id, **kwargs)
            return [os.path.join(kwargs["directory"], f"{order_id}_{i}.tif")
                    for i in range(2)]

        def list_orders(self):
            rec.add("orders.list_orders")
            orders = [{"id": "o1"}, {"id": "o2"}]
            if list_mode == "aiter":
                async def _gen():
                    for order in orders:
                        yield order
                return _gen()
            if list_mode == "await":
                async def _coro():
                    return tuple(orders)
                return _coro()
            return iter(orders)

        def cancel_orders(self, order_ids=None):
            rec.add("orders.cancel_orders", order_ids=order_ids)
            res = {"result": {"succeeded": {"count": len(order_ids or [])}}}
            if cancel_awaitable:
                async def _coro():
                    return res
                return _coro()
            return res

    class _Session:
        def __init__(self, auth=None):
            self.auth = auth
            self.clients = []
            rec.sessions.append(self)
            self.open = False

        async def __aenter__(self):
            self.open = True
            return self

        async def __aexit__(self, *_args):
            self.open = False
            return False

        def client(self, name):
            assert self.open, "client() used outside the session context"
            self.clients.append(name)
            return {"data": _DataClient, "orders": _OrdersClient}[name]()

    return types.SimpleNamespace(
        data_filter=data_filter, order_request=order_request, Session=_Session
    )


@pytest.fixture(name="fake")
def fixture_fake(monkeypatch):
    def _install(**kwargs):
        rec = _Recorder()
        monkeypatch.setattr(pl, "PLANET_AVAIL", True)
        monkeypatch.setattr(pl, "planet", _make_fake_planet(rec, **kwargs), raising=False)
        return rec
    return _install


def _assert_one_session(rec, client):
    assert len(rec.sessions) == 1
    assert rec.sessions[0].auth is AUTH
    assert rec.sessions[0].clients == [client]
    assert not rec.sessions[0].open  # closed again


def test_run_search_planet_items_all_filters(fake):
    rec = fake(n_items=5)
    start = datetime.datetime(2024, 6, 1)
    end = datetime.datetime(2024, 6, 30)
    items = pl.run_search_planet_items(
        AUTH, pl.RSGIS_PLANET_ITEM_SkySatScene, (-4.1, -4.0, 52.4, 52.5),
        start_date=start, end_date=end, cloud_cover=10.0,
        sun_elevation_min=20.0, sun_elevation_max=70.0,
        view_angle_min=0.0, view_angle_max=5.0, max_n_rslts=4,
    )
    assert items == [{"id": f"item{i}"} for i in range(4)]  # limited to 4
    _assert_one_session(rec, "data")

    search = rec.get("data.search")[2]
    assert search["item_types"] == ["SkySatScene"]
    assert search["sort"] == "acquired asc"
    assert search["limit"] == 4
    # The search filter is an AND of all the individual filters, in order.
    and_args = rec.get("filter.and")[1][0]
    assert search["search_filter"][0] == "and"
    assert [f[0] for f in and_args] == [
        "permission", "geometry", "date_range", "range", "range", "range",
    ]
    geom = rec.get("filter.geometry")[1][0]
    assert geom["type"] == "Polygon"
    xs = [pt[0] for pt in geom["coordinates"][0]]
    ys = [pt[1] for pt in geom["coordinates"][0]]
    assert (min(xs), max(xs), min(ys), max(ys)) == (-4.1, -4.0, 52.4, 52.5)
    assert rec.get("filter.date_range")[1:] == (("acquired",), {"gte": start, "lte": end})
    ranges = [c[1:] for c in rec.calls if c[0] == "filter.range"]
    assert ranges == [
        (("cloud_percent",), {"lte": 10.0}),
        (("sun_elevation",), {"gte": 20.0, "lte": 70.0}),
        (("view_angle",), {"gte": 0.0, "lte": 5.0}),
    ]


def test_run_search_planet_items_minimal_filters(fake):
    rec = fake(n_items=2)
    items = pl.run_search_planet_items(
        AUTH, pl.RSGIS_PLANET_ITEM_PSScene, [0.0, 1.0, 50.0, 51.0]
    )
    assert [i["id"] for i in items] == ["item0", "item1"]
    search = rec.get("data.search")[2]
    assert search["item_types"] == ["PSScene"] and search["limit"] == 25
    # Only the permission and geometry filters when nothing else is given;
    # a start date on its own still adds the date filter (open ended).
    assert [f[0] for f in rec.get("filter.and")[1][0]] == ["permission", "geometry"]
    rec2 = fake()
    start = datetime.datetime(2024, 1, 1)
    pl.run_search_planet_items(AUTH, pl.RSGIS_PLANET_ITEM_PSScene,
                               (0.0, 1.0, 50.0, 51.0), start_date=start)
    assert rec2.get("filter.date_range")[2] == {"gte": start, "lte": None}
    with pytest.raises(rsgispy.RSGISPyException):
        pl.run_search_planet_items(AUTH, 999, (0.0, 1.0, 50.0, 51.0))


def test_run_create_planet_order(fake):
    rec = fake()
    items = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    order = pl.run_create_planet_order(
        AUTH, "my_order", items, pl.RSGIS_PLANET_ITEM_PSScene,
        pl.RSGIS_PLANET_BUNDLE_analytic_sr_udm2, email_notification=False,
    )
    _assert_one_session(rec, "orders")
    assert rec.get("order.product")[2] == {
        "item_ids": ["a", "b", "c"], "product_bundle": "analytic_sr_udm2",
        "item_type": "PSScene",
    }
    assert rec.get("order.notifications")[2] == {"email": False}
    req = rec.get("order.build_request")[2]
    assert req["name"] == "my_order"
    assert req["products"] == [{"product": rec.get("order.product")[2]}]
    # The request that was built is the one sent, and the order is returned.
    sent = rec.get("orders.create_order")[1][0]
    assert sent == {"request": req}
    assert order["id"] == "order-1" and order["request"] is sent
    with pytest.raises(rsgispy.RSGISPyException):
        pl.run_create_planet_order(AUTH, "x", items, pl.RSGIS_PLANET_ITEM_PSScene, 999)


def test_run_download_planet_order(fake, tmp_path):
    rec = fake(order_state="success")
    files = pl.run_download_planet_order(AUTH, "oid-1", str(tmp_path), overwrite=True)
    assert files == [os.path.join(str(tmp_path), f"oid-1_{i}.tif") for i in range(2)]
    _assert_one_session(rec, "orders")
    assert rec.names() == ["orders.get_order", "orders.download_order"]
    assert rec.get("orders.download_order")[1:] == (
        ("oid-1",),
        {"directory": str(tmp_path), "overwrite": True, "progress_bar": True},
    )

    rec = fake(order_state="running")
    with pytest.raises(rsgispy.RSGISPyException, match="not yet"):
        pl.run_download_planet_order(AUTH, "oid-2", str(tmp_path))
    assert rec.names() == ["orders.get_order"]  # nothing downloaded


@pytest.mark.parametrize("list_mode", ["aiter", "await", "iter"])
def test_run_get_planet_orders(fake, list_mode):
    rec = fake(list_mode=list_mode)
    orders = pl.run_get_planet_orders(AUTH)
    assert orders == [{"id": "o1"}, {"id": "o2"}]
    assert isinstance(orders, list)
    _assert_one_session(rec, "orders")


@pytest.mark.parametrize("cancel_awaitable", [True, False])
def test_run_cancel_planet_orders(fake, cancel_awaitable):
    rec = fake(cancel_awaitable=cancel_awaitable)
    res = pl.run_cancel_planet_orders(AUTH, order_ids=["a", "b"])
    assert res == {"result": {"succeeded": {"count": 2}}}
    assert rec.get("orders.cancel_orders")[2] == {"order_ids": ["a", "b"]}
    _assert_one_session(rec, "orders")
    rec = fake(cancel_awaitable=cancel_awaitable)
    pl.run_cancel_planet_orders(AUTH)  # None = cancel all active orders
    assert rec.get("orders.cancel_orders")[2] == {"order_ids": None}


def test_run_functions_need_planet(monkeypatch, tmp_path):
    monkeypatch.setattr(pl, "PLANET_AVAIL", False)
    calls = [
        lambda: pl.run_search_planet_items(AUTH, 1, (0, 1, 0, 1)),
        lambda: pl.run_download_and_validate_item(AUTH, 1, "id", 1, str(tmp_path)),
        lambda: pl.run_create_planet_order(AUTH, "n", [{"id": "a"}], 1, 1),
        lambda: pl.run_download_planet_order(AUTH, "id", str(tmp_path)),
        lambda: pl.run_get_planet_orders(AUTH),
        lambda: pl.run_cancel_planet_orders(AUTH),
    ]
    for call in calls:
        with pytest.raises(rsgispy.RSGISPyException, match="pip install planet"):
            call()


def test_run_functions_refuse_inside_running_loop(fake, tmp_path):
    rec = fake()

    async def _inside_loop():
        with pytest.raises(rsgispy.RSGISPyException, match="event loop"):
            pl.run_get_planet_orders(AUTH)
        with pytest.raises(rsgispy.RSGISPyException, match="event loop"):
            pl.run_cancel_planet_orders(AUTH, ["a"])
        with pytest.raises(rsgispy.RSGISPyException, match="event loop"):
            pl.run_download_planet_order(AUTH, "id", str(tmp_path))

    asyncio.run(_inside_loop())
    assert not rec.sessions  # the coroutines were closed, never started
