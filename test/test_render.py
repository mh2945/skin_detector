"""오버레이 렌더 검증. 웹 결과 화면이 실제로 그려지는 경로다.

`render.py` 는 cv2 를 쓰므로 `core/` 밖이다 (계약 1번). 여기서 검사하는 것은
색 과학이 아니라 **그림이 어긋나지 않는가**다:

- 히트맵이 tau 아래에서 완전히 투명한가 (아니면 깨끗한 피부가 붉게 칠해진다)
- 마스크 밖을 칠하지 않는가
- 게이트 실패 시 아무것도 그리지 않는가 (못 잰 것을 그림으로 보여주면 안 된다)
"""

import numpy as np
import pytest

from conftest import (ROOT, disc, hemoglobin_direction, inject,
                      synthetic_face_linear, synthetic_observation)
from skin_detector import imageio as sio, pipeline, render

CLEAN = dict(texture=0.05, noise=0.004)


@pytest.fixture(scope="module")
def cfg_pair():
    cfg = sio.load_config()
    return cfg, sio.config_hash(cfg)


def analyze(linear, cfg_pair, **kw):
    cfg, chash = cfg_pair
    return pipeline.analyze(np.zeros((4, 4, 3), np.uint8), cfg, chash,
                            model_path=None,
                            observation=synthetic_observation(linear, **kw))


@pytest.fixture(scope="module")
def dosed(cfg_pair):
    """오른뺨에 홍반이 있는, 게이트를 통과한 분석 결과."""
    base = synthetic_face_linear(**CLEAN)
    blob = disc(250.0, 430.0, 90.0, 0.06)
    ana = analyze(inject(base, blob, hemoglobin_direction()), cfg_pair)
    assert ana.result.quality.passed
    return ana


def test_heatmap_is_transparent_below_tau(dosed):
    """tau 아래는 **완전히** 투명해야 한다.

    조금이라도 칠하면 자기참조 기준선상 항상 존재하는 잔여 편차가 화면에서
    '옅은 붉은기'로 보인다 — 사용자는 그걸 증상으로 읽는다.
    """
    tau = dosed.result.baseline.tau
    rgb, alpha = render.heatmap(dosed.deviation, dosed.skin, tau)

    assert rgb.shape == dosed.deviation.shape + (3,)
    assert alpha.shape == dosed.deviation.shape
    below = dosed.skin & (dosed.deviation <= tau)
    assert float(alpha[below].max()) == 0.0
    assert float(alpha.max()) > 0.0, "홍반을 넣었으니 어딘가는 칠해져야 한다"
    assert float(alpha.max()) <= 1.0


def test_heatmap_never_paints_outside_the_skin_mask(dosed):
    tau = dosed.result.baseline.tau
    _rgb, alpha = render.heatmap(dosed.deviation, dosed.skin, tau)
    assert float(alpha[~dosed.skin].max()) == 0.0


def test_render_overlay_returns_a_drawable_image(dosed):
    img = render.render_overlay(dosed)
    assert img is not None
    assert img.dtype == np.uint8
    assert img.shape == dosed.observation.canonical_rgb.shape
    # 원본과 달라야 한다 — 뭔가 그려졌다는 뜻이다.
    assert not np.array_equal(img, dosed.observation.canonical_rgb)


def test_render_overlay_variants_do_not_crash(dosed):
    for regions, lesions in ((True, True), (True, False), (False, True), (False, False)):
        img = render.render_overlay(dosed, show_regions=regions, show_lesions=lesions)
        assert img is not None and img.dtype == np.uint8


def test_render_overlay_is_none_without_an_observation(cfg_pair):
    """얼굴을 못 찾았으면 그릴 것이 없다 — 빈 그림이 아니라 `None` 이어야 한다."""
    model = ROOT / "models" / "face_landmarker.task"
    if not model.exists():
        pytest.skip("랜드마커 모델 번들이 없다 (scripts/setup_env.ps1)")
    cfg, chash = cfg_pair
    flat = np.full((480, 640, 3), 128, dtype=np.uint8)      # 얼굴이 아니다
    ana = pipeline.analyze(flat, cfg, chash, model_path=model, observation=None)
    assert ana.observation is None
    assert ana.result.quality.reason.value == "no_face"
    assert render.render_overlay(ana) is None


def test_save_png_roundtrip(dosed, tmp_path):
    import cv2
    out = tmp_path / "overlay.png"
    render.save_png(out, render.render_overlay(dosed))
    assert out.exists() and out.stat().st_size > 0
    # save_png 는 RGB->BGR 로 뒤집어 저장한다. 되읽어 다시 뒤집으면 원본이어야 한다.
    back = cv2.imread(str(out))[:, :, ::-1]
    assert np.array_equal(back, render.render_overlay(dosed))


def test_contact_sheet_tiles_every_image(dosed):
    img = render.render_overlay(dosed)
    sheet = render.contact_sheet([img, img, img], cols=2, tile=64)
    assert sheet.dtype == np.uint8
    # 3장을 2열로 -> 2행
    assert sheet.shape[0] >= 64 * 2
    assert sheet.shape[1] >= 64 * 2


def test_contact_sheet_of_nothing_is_harmless():
    """빈 목록에도 죽지 않는다. store 가 비어 있을 때 tune.py 가 지나가는 경로다."""
    sheet = render.contact_sheet([], cols=4, tile=64)
    assert sheet is not None and sheet.dtype == np.uint8
    assert not sheet.any(), "그릴 것이 없으면 빈 시트여야 한다"
