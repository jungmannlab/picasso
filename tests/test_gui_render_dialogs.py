"""Render's dialogs and interactive flows, run as a user would, with the
modal dialogs answered by the test.

- selecting picks one by one by their scatter (2D, 3D and with
  projections) or by their time traces;
- the custom colormap editor;
- Imaris export of the field of view (the writer is stubbed: it exists
  on Windows only);
- applying expressions to the localizations;
- reopening rotated localizations saved by the 3D window;
- G5M from the menu;
- the parameter dialogs of clustering, linking, AIM, CBC, spatial
  statistics and manual export, filled in rather than bypassed;
- saving (picked) localizations, exporting images, masking, dropping
  files onto the window, the Files dialog and other Tools entries.

The workers of cancelable tasks and of ``lib.run_with_status`` run on a
Qt thread, which coverage does not trace; the files and the state these
tests check are their result.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PyQt6 import QtWidgets

from picasso.gui import render as gui_render
from tests.test_gui_render_analysis import (
    APPLY_TO_ALL,
    _Answers,
    _saved,
    _window,
    _window_3d,
)
from tests.test_gui_render_tasks import _cluster_centers, _wait_for_tasks


@pytest.fixture
def answers(monkeypatch):
    return _Answers(monkeypatch)


@pytest.fixture
def window(qt_offscreen, tmp_path):
    window = _window(tmp_path)
    yield window
    _wait_for_tasks(qt_offscreen)
    window.close()


@pytest.fixture
def window2(qt_offscreen, tmp_path):
    """Two channels."""
    window = _window(tmp_path, n_channels=2)
    yield window
    _wait_for_tasks(qt_offscreen)
    window.close()


@pytest.fixture
def window_3d(qt_offscreen, tmp_path):
    """Two channels of 3D localizations."""
    window = _window_3d(tmp_path)
    yield window
    _wait_for_tasks(qt_offscreen)
    window.close()


# ---------------------------------------------------------------------------
# Selecting picks one by one
# ---------------------------------------------------------------------------


def _pick_replies(monkeypatch, *replies) -> list[str]:
    """Answer the boxes of ``View.pick_message_box`` by clicking the
    buttons named in ``replies``; returns the texts shown."""
    replies = list(replies)
    shown = []

    def exec_(box):
        shown.append(box.text())
        reply = replies.pop(0)
        (button,) = [b for b in box.buttons() if b.text() == reply]
        button.click()
        return 0

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", exec_)
    return shown


def _figures() -> set[int]:
    """The open pyplot figures."""
    return set(plt.get_fignums())


def _circle_picks(window, n: int = 3, diameter: float = 300.0) -> list:
    """Circular picks around the first ``n`` clusters."""
    view = window.view
    view._pick_shape = "Circle"
    window.tools_settings_dialog.pick_diameter.setValue(diameter)  # nm
    picks = list(_cluster_centers()[:n])
    view._picks = list(picks)
    view.n_picks = n
    return picks


def test_scatter_selection_removes_rejected_picks(window, monkeypatch):
    picks = _circle_picks(window)
    shown = _pick_replies(monkeypatch, "Accept", "Reject", "Accept")
    window.view.show_pick()
    assert window.view._picks == [picks[0], picks[2]]
    assert window.view.n_picks == 2
    assert shown[0].startswith("Keep pick No: 1 of 3 ?")
    assert "Picks removed: 1 Picks kept: 1" in shown[2]
    assert window.info_dialog.n_picks.text() == "2"


def test_scatter_selection_back_and_accept_keeps_the_pick(window, monkeypatch):
    picks = _circle_picks(window)
    shown = _pick_replies(
        monkeypatch, "Reject", "Back", "Accept", "Accept", "Accept"
    )
    window.view.show_pick()
    assert window.view._picks == picks
    assert [text.split("\n")[0] for text in shown[:3]] == [
        "Keep pick No: 1 of 3 ?",
        "Keep pick No: 2 of 3 ?",
        "Keep pick No: 1 of 3 ?",
    ]


def test_scatter_selection_rejecting_a_pick_twice_removes_it_once(
    window, monkeypatch
):
    """Going back to a rejected pick and rejecting it again listed it
    twice for removal; removing it the second time raised
    ValueError."""
    picks = _circle_picks(window)
    _pick_replies(monkeypatch, "Reject", "Back", "Reject", "Accept", "Accept")
    window.view.show_pick()
    assert window.view._picks == picks[1:]


def test_scatter_selection_cancel_keeps_the_rest(window, monkeypatch):
    picks = _circle_picks(window)
    _pick_replies(monkeypatch, "Reject", "Cancel")
    window.view.show_pick()
    assert window.view._picks == picks[1:]


def test_scatter_selection_of_all_channels(window2, answers, monkeypatch):
    picks = _circle_picks(window2, n=2)
    answers.item("Show all channels")
    _pick_replies(monkeypatch, "Reject", "Accept")
    figures = _figures()
    window2.view.show_pick()
    assert window2.view._picks == picks[1:]
    assert _figures() == figures


def test_scatter_selection_canceled_channel_keeps_the_picks(
    window2, answers, monkeypatch
):
    picks = _circle_picks(window2, n=2)
    answers.item(None)
    shown = _pick_replies(monkeypatch)
    window2.view.show_pick()
    assert shown == []
    assert window2.view._picks == picks


@pytest.mark.parametrize(
    "method",
    ["show_pick", "show_pick_3d", "show_pick_3d_iso", "select_traces"],
)
def test_pick_selection_needs_a_pick(method, window, answers):
    answers.record_messages()
    getattr(window.view, method)()
    assert answers.messages == ["No localizations picked. Please pick first."]


def test_trace_selection_removes_rejected_picks(window, monkeypatch):
    picks = _circle_picks(window)
    shown = _pick_replies(monkeypatch, "Reject", "Accept", "Reject")
    figures = _figures()
    window.view.select_traces()
    assert window.view._picks == [picks[1]]
    assert len(shown) == 3
    assert _figures() == figures


@pytest.mark.parametrize("cancel_at", [0, 1])
def test_trace_selection_cancel_closes_the_trace(
    cancel_at, window, monkeypatch
):
    picks = _circle_picks(window)
    _pick_replies(monkeypatch, *(["Reject"] * cancel_at + ["Cancel"]))
    figures = _figures()
    window.view.select_traces()
    assert window.view._picks == picks[cancel_at:]
    assert _figures() == figures


def test_trace_selection_back_and_reject_twice(window, monkeypatch):
    picks = _circle_picks(window, n=2)
    _pick_replies(monkeypatch, "Accept", "Back", "Reject", "Reject")
    window.view.select_traces()
    assert window.view._picks == []
    _circle_picks(window, n=2)
    _pick_replies(monkeypatch, "Reject", "Back", "Reject", "Accept")
    window.view.select_traces()
    assert window.view._picks == picks[1:]


def _plot_dialog_replies(monkeypatch, dialog_class, *replies) -> list:
    """Answer ``dialog_class`` (``PlotDialog`` or ``PlotDialogIso``) by
    clicking Yes ("yes"), No ("no") or Cancel ("cancel"); returns the
    labels and the titles of the plotted axes."""
    replies = list(replies)
    shown = []
    buttons = QtWidgets.QDialogButtonBox.StandardButton
    names = {"yes": buttons.Yes, "no": buttons.No, "cancel": buttons.Cancel}

    def exec_(dialog):
        axes = dialog.figure.axes
        shown.append(
            (dialog.label.text(), [ax.get_title() for ax in axes], axes)
        )
        reply = replies.pop(0)
        if reply == "escape":
            dialog.reject()  # what Escape and the close button do
        else:
            dialog.buttons.button(names[reply]).click()
        return 0

    monkeypatch.setattr(dialog_class, "exec", exec_)
    return shown


@pytest.mark.parametrize(
    "method, dialog_class, titles",
    [
        ("show_pick_3d", gui_render.PlotDialog, [""]),
        (
            "show_pick_3d_iso",
            gui_render.PlotDialogIso,
            ["3D", "XY", "XZ", "YZ"],
        ),
    ],
)
@pytest.mark.parametrize("all_channels", [False, True])
def test_3d_selection_removes_rejected_picks(
    method, dialog_class, titles, all_channels, window_3d, answers, monkeypatch
):
    view = window_3d.view
    view._pick_shape = "Circle"
    window_3d.tools_settings_dialog.pick_diameter.setValue(1000.0)
    picks = [(1000.0, 1000.0), (2000.0, 2000.0), (3000.0, 3000.0)]
    view._picks = list(picks)
    answers.item("Show all channels" if all_channels else view.locs_paths[1])
    shown = _plot_dialog_replies(
        monkeypatch, dialog_class, "no", "yes", "cancel"
    )
    figures = _figures()
    getattr(view, method)()
    assert view._picks == picks[1:]
    assert [label for label, _, _ in shown] == [
        f"3D Scatterplot of pick {i} of 3." for i in (1, 2, 3)
    ]
    label, axes_titles, axes = shown[0]
    if all_channels:  # the two channels in one plot
        assert len(axes[0].collections) == 2
        expected = [t if t != "3D" else "" for t in titles]
    else:  # colored by z
        assert len(axes[0].collections) == 1
        expected = titles
    assert axes_titles == expected
    assert _figures() == figures


@pytest.mark.parametrize(
    "method, dialog_class",
    [
        ("show_pick_3d", gui_render.PlotDialog),
        ("show_pick_3d_iso", gui_render.PlotDialogIso),
    ],
)
def test_closing_the_3d_selection_keeps_the_picks(
    method, dialog_class, window_3d, answers, monkeypatch
):
    """Escape or the close button stop the selection, as Cancel; the
    pick shown is not rejected."""
    view = window_3d.view
    view._pick_shape = "Circle"
    window_3d.tools_settings_dialog.pick_diameter.setValue(1000.0)
    picks = [(1000.0, 1000.0), (2000.0, 2000.0)]
    view._picks = list(picks)
    answers.item(view.locs_paths[1])
    shown = _plot_dialog_replies(monkeypatch, dialog_class, "escape")
    getattr(view, method)()
    assert len(shown) == 1
    assert view._picks == picks


# ---------------------------------------------------------------------------
# Custom colormap editor
# ---------------------------------------------------------------------------


def _names(dialog) -> list[str]:
    names = dialog.name_list
    return [names.item(row).text() for row in range(names.count())]


def _table(dialog) -> list[list[str]]:
    table = dialog.stops_table
    return [
        [table.item(row, col).text() for col in range(4)]
        for row in range(table.rowCount())
    ]


@pytest.fixture
def cmap_dialog(window, answers):
    """The custom colormap editor with one colormap, "Fire"."""
    answers.record_messages()
    window.custom_colormaps_stops = {
        "Fire": [[0.0, 0.0, 0.0, 0.0], [0.5, 1.0, 0.0, 0.0], [1.0, 1, 1, 0]]
    }
    dialog = gui_render.CustomColormapDialog(window)
    yield dialog
    dialog.close()


def _stored_colormaps() -> dict:
    from picasso import io

    return io.load_user_settings()["Render"]["CustomColormaps"]


def test_colormap_editor_without_colormaps(window):
    del window.custom_colormaps_stops
    dialog = gui_render.CustomColormapDialog(window)
    assert window.custom_colormaps_stops == {}
    assert _names(dialog) == []
    assert not dialog.stops_table.isEnabled()
    assert not dialog.delete_btn.isEnabled()
    for button in (dialog.dup_btn, dialog.rename_btn, dialog.delete_btn):
        button.click()  # nothing selected: nothing happens
    dialog.add_stop_btn.click()
    dialog.remove_stop_btn.click()
    assert window.custom_colormaps_stops == {}
    dialog.close()


def test_colormap_editor_opens_from_the_files_dialog(window, monkeypatch):
    opened = []
    monkeypatch.setattr(
        gui_render.CustomColormapDialog,
        "exec",
        lambda dialog: opened.append(_names(dialog)) or 0,
    )
    window.dataset_dialog.open_custom_cmap_editor()
    assert opened == [[]]


def test_colormap_editor_shows_the_first_colormap(cmap_dialog):
    assert _names(cmap_dialog) == ["Fire"]
    assert cmap_dialog.stops_table.isEnabled()
    assert _table(cmap_dialog)[1] == ["0.500", "1.000", "0.000", "0.000"]
    assert not cmap_dialog.preview_label.pixmap().isNull()


def test_new_colormap_is_saved_and_offered_for_the_channels(
    cmap_dialog, window, answers
):
    answers.text("  Ice ")
    cmap_dialog.new_btn.click()
    assert _names(cmap_dialog) == ["Fire", "Ice"]
    assert cmap_dialog.name_list.currentItem().text() == "Ice"
    assert window.custom_colormaps_stops["Ice"] == [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    assert _stored_colormaps()["Ice"] == [[0.0] * 4, [1.0] * 4]
    combo = window.dataset_dialog.colorselection[0]
    assert combo.findText("Ice") >= 0


@pytest.mark.parametrize(
    "name, error",
    [
        ("", "Name must not be empty."),
        ("red", "'red' collides with a built-in solid color."),
        ("red_gradient", "'red_gradient' collides with a built-in colormap."),
        ("Fire", "A custom colormap named 'Fire' already exists."),
    ],
)
def test_new_colormap_name_must_be_valid(name, error, cmap_dialog, answers):
    answers.text(name, "Ice")
    cmap_dialog.new_btn.click()
    assert answers.messages == [error]
    assert _names(cmap_dialog) == ["Fire", "Ice"]


def test_new_colormap_canceled(cmap_dialog, window, answers):
    answers.text(None)
    cmap_dialog.new_btn.click()
    assert list(window.custom_colormaps_stops) == ["Fire"]


def test_duplicate_colormap_suggests_a_unique_name(
    cmap_dialog, window, monkeypatch
):
    suggested = []

    def get_text(*args, text="", **kwargs):
        suggested.append(text)
        return text, True

    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", get_text)
    cmap_dialog.dup_btn.click()
    cmap_dialog.name_list.setCurrentRow(0)
    cmap_dialog.dup_btn.click()
    assert suggested == ["Fire_copy", "Fire_copy_2"]
    stops = window.custom_colormaps_stops
    assert stops["Fire_copy_2"] == stops["Fire"]
    assert stops["Fire_copy"] is not stops["Fire"]  # a copy


def test_rename_colormap(cmap_dialog, window, monkeypatch):
    stops = window.custom_colormaps_stops["Fire"]
    names = ["Fire", "Lava"]
    suggested = []

    def get_text(*args, text="", **kwargs):
        suggested.append(text)
        return names.pop(0), True

    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", get_text)
    cmap_dialog.rename_btn.click()  # the same name: nothing to do
    assert suggested == ["Fire"]  # not "Fire_2"
    assert _names(cmap_dialog) == ["Fire"]
    cmap_dialog.rename_btn.click()
    assert _names(cmap_dialog) == ["Lava"]
    assert window.custom_colormaps_stops == {"Lava": stops}
    assert list(_stored_colormaps()) == ["Lava"]


@pytest.mark.parametrize("confirm", [False, True])
def test_delete_colormap_asks_first(confirm, cmap_dialog, window, monkeypatch):
    button = QtWidgets.QMessageBox.StandardButton
    asked = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *a, **k: asked.append(a[2])
        or (button.Yes if confirm else button.No),
    )
    cmap_dialog.delete_btn.click()
    assert asked == ["Delete custom colormap 'Fire'?"]
    assert ("Fire" in window.custom_colormaps_stops) is not confirm
    if confirm:
        assert _names(cmap_dialog) == []
        assert not cmap_dialog.stops_table.isEnabled()
        assert _stored_colormaps() == {}


def test_delete_colormap_selects_the_next_one(
    cmap_dialog, window, answers, monkeypatch
):
    answers.text("Ice")
    cmap_dialog.new_btn.click()
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    cmap_dialog.delete_btn.click()  # Ice
    assert _names(cmap_dialog) == ["Fire"]
    assert cmap_dialog.name_list.currentRow() == 0
    assert cmap_dialog.stops_table.rowCount() == 3


def test_add_stops_up_to_the_limit(cmap_dialog, window, answers):
    cmap_dialog.add_stop_btn.click()
    # in the middle of the last two stops
    assert _table(cmap_dialog)[2] == ["0.750", "0.500", "0.500", "0.500"]
    cmap_dialog.add_stop_btn.click()
    assert cmap_dialog.stops_table.rowCount() == 5
    cmap_dialog.add_stop_btn.click()
    assert answers.messages == ["At most 5 stops are supported."]
    assert len(window.custom_colormaps_stops["Fire"]) == 5
    assert len(_stored_colormaps()["Fire"]) == 5


def test_remove_stop_keeps_the_first_and_last(cmap_dialog, window, answers):
    table = cmap_dialog.stops_table
    for row in (-1, 0, 2):
        table.setCurrentCell(row, 0)
        cmap_dialog.remove_stop_btn.click()
    assert answers.messages == ["The first and last stops are required."] * 3
    table.setCurrentCell(1, 0)
    cmap_dialog.remove_stop_btn.click()
    assert window.custom_colormaps_stops["Fire"] == [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
    ]
    cmap_dialog.remove_stop_btn.click()
    assert answers.messages[-1] == "At least 2 stops are required."


@pytest.mark.parametrize(
    "text, shown", [("2", "1.000"), ("-1", "0.000"), ("abc", "0.000")]
)
def test_edited_stop_is_clamped_and_saved(text, shown, cmap_dialog, window):
    cmap_dialog.stops_table.item(1, 2).setText(text)
    assert cmap_dialog.stops_table.item(1, 2).text() == shown
    assert window.custom_colormaps_stops["Fire"][1][2] == float(shown)
    assert _stored_colormaps()["Fire"][1][2] == float(shown)


def test_invalid_stop_positions_clear_the_preview(cmap_dialog, window):
    cmap_dialog.stops_table.item(1, 0).setText("0")  # not increasing
    assert cmap_dialog.preview_label.pixmap().isNull()
    cmap_dialog.stops_table.item(1, 0).setText("0.3")
    assert not cmap_dialog.preview_label.pixmap().isNull()


def test_editing_a_colormap_recolors_the_channel(cmap_dialog, window):
    dataset = window.dataset_dialog
    dataset.auto_colors.setChecked(False)
    dataset.colorselection[0].setCurrentText("Fire")
    sample = gui_render.LEGEND_SAMPLE_IDX
    before = dataset._channel_luts[0][sample].copy()
    cmap_dialog.stops_table.item(1, 3).setText("1")  # red to magenta
    after = dataset._channel_luts[0][sample]
    assert after[2] > before[2]
    assert dataset.colorselection[0].currentText() == "Fire"


def test_stop_color_from_the_color_picker(cmap_dialog, window, monkeypatch):
    from PyQt6 import QtGui

    colors = [QtGui.QColor(0, 0, 255), QtGui.QColor()]  # 2nd: canceled
    asked = []

    def get_color(initial, *args):
        asked.append(initial.getRgbF()[:3])
        return colors.pop(0)

    monkeypatch.setattr(QtWidgets.QColorDialog, "getColor", get_color)
    cmap_dialog._on_cell_double_clicked(1, 0)  # the position: no picker
    cmap_dialog._on_cell_double_clicked(1, 1)
    assert asked == [(1.0, 0.0, 0.0)]
    assert _table(cmap_dialog)[1] == ["0.500", "0.000", "0.000", "1.000"]
    assert window.custom_colormaps_stops["Fire"][1] == [0.5, 0.0, 0.0, 1.0]
    cmap_dialog._on_cell_double_clicked(1, 2)
    assert window.custom_colormaps_stops["Fire"][1] == [0.5, 0.0, 0.0, 1.0]


def test_colormap_that_cannot_be_saved_warns(
    cmap_dialog, window, answers, monkeypatch
):
    def fail(settings):
        raise OSError("disk full")

    answers.text("Ice")
    with monkeypatch.context() as patch:
        patch.setattr(gui_render.io, "save_user_settings", fail)
        cmap_dialog.new_btn.click()
    assert "Ice" in window.custom_colormaps_stops  # kept for the session
    assert answers.messages == [
        "Failed to persist custom colormaps to disk:\ndisk full"
    ]


def test_saved_colormaps_are_loaded_with_the_settings(window):
    from picasso import io

    settings = io.load_user_settings()
    settings["Render"]["CustomColormaps"] = {
        "Ice": [[0, 0, 0, 0], [1, 0, 1, 1]],
        "broken": [[0, 0, 0], ["a", 1, 1, 1]],
        "none": None,
    }
    io.save_user_settings(settings)
    window.load_user_settings()
    assert window.custom_colormaps_stops == {
        "Ice": [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 1.0]]
    }
    assert window.dataset_dialog.colorselection[0].findText("Ice") >= 0


# ---------------------------------------------------------------------------
# Imaris export of the field of view
# ---------------------------------------------------------------------------


@pytest.fixture
def imaris(monkeypatch):
    """Stub the Imaris writer (Windows only); records its calls."""
    from types import SimpleNamespace

    written = []
    monkeypatch.setattr(
        gui_render, "PW", SimpleNamespace(Color=lambda *c: c), raising=False
    )

    def numpy_to_imaris(image, path, colors, disp_px_size, viewport, *args):
        written.append(
            SimpleNamespace(
                image=image,
                path=path,
                colors=colors,
                disp_px_size=disp_px_size,
                z_range=args[-2:],
            )
        )
        with open(path, "w") as f:
            f.write("ims")

    monkeypatch.setattr(
        gui_render, "numpy_to_imaris", numpy_to_imaris, raising=False
    )
    return written


def test_imaris_export_of_3d_localizations(
    window_3d, imaris, answers, tmp_path
):
    out = tmp_path / "roi.ims"
    answers.save_path(out)
    window_3d.export_fov_ims()
    (written,) = imaris
    assert written.path == str(out)
    assert written.image.ndim == 4  # channel, z, and the image
    assert written.image.shape[0] == len(written.colors) == 2
    z = np.concatenate([locs["z"] for locs in window_3d.view.locs])
    assert written.z_range == pytest.approx((z.min(), z.max()), abs=1.0)
    assert _saved(tmp_path) == [
        "roi.ims",
        "roi_ch_0.hdf5",
        "roi_ch_0.yaml",
        "roi_ch_1.hdf5",
        "roi_ch_1.yaml",
    ]


def test_imaris_export_replaces_the_file(window, imaris, answers, tmp_path):
    out = tmp_path / "roi.ims"
    out.write_text("old")
    answers.save_path(out)
    window.export_fov_ims()
    assert out.read_text() == "ims"


def test_imaris_export_saturates_bright_pixels(
    window, imaris, answers, tmp_path
):
    """Pixels brighter than the contrast maximum wrapped around in the
    16-bit image and showed up dark."""
    window.display_settings_dlg.maximum.setValue(1.0)
    answers.save_path(tmp_path / "roi.ims")
    window.export_fov_ims()
    image = imaris[0].image
    assert image.max() == 65535
    assert set(np.unique(image)) == {0, 65535}


def test_imaris_export_needs_localizations(qt_offscreen, answers):
    answers.record_messages()
    window = gui_render.Window(plugins_loaded=True)
    window.export_fov_ims()
    assert len(answers.messages) == 1
    window.close()


def test_imaris_export_needs_a_shown_channel(
    window, imaris, answers, tmp_path
):
    answers.record_messages()
    window.dataset_dialog.checks[0].setChecked(False)
    answers.save_path(tmp_path / "roi.ims")
    window.export_fov_ims()
    assert imaris == []
    assert answers.messages == ["Show at least one channel to export."]
    assert _saved(tmp_path) == []


def test_imaris_export_refuses_mixed_2d_and_3d(
    qt_offscreen, imaris, answers, tmp_path
):
    from tests.test_gui_render_tasks import _info, _locs

    answers.record_messages()
    window = _window_3d(tmp_path)
    window.view.add(
        str(tmp_path / "flat.hdf5"), _locs(), _info(), render_=False
    )
    answers.save_path(tmp_path / "roi.ims")
    window.export_fov_ims()
    assert imaris == []
    assert answers.messages == [
        "Cannot export localizations with and without z together."
    ]
    assert _saved(tmp_path) == []
    window.close()


# ---------------------------------------------------------------------------
# Apply expression
# ---------------------------------------------------------------------------


def _apply(window, monkeypatch, cmd: str, channel: str | None = None, ok=True):
    """Apply ``cmd`` from the Apply expression dialog to ``channel``
    (its name in the dialog; the first one if None)."""
    shown = []

    def exec_(dialog):
        shown.append(dialog.label.text())
        dialog.cmd.setText(cmd)
        if channel is not None:
            dialog.channel.setCurrentText(channel)
        accepted = QtWidgets.QDialog.DialogCode.Accepted
        rejected = QtWidgets.QDialog.DialogCode.Rejected
        return accepted if ok else rejected

    monkeypatch.setattr(gui_render.ApplyDialog, "exec", exec_)
    window.open_apply_dialog()
    return shown


def _xy(view, channel: int = 0) -> np.ndarray:
    return view.locs[channel][["x", "y"]].to_numpy(dtype=float)


def test_apply_expression_moves_the_localizations(window, monkeypatch):
    before = _xy(window.view)
    shown = _apply(window, monkeypatch, "x += 100")
    assert "'photons'" in shown[0]  # the variables
    after = _xy(window.view)
    np.testing.assert_allclose(after[:, 0], before[:, 0] + 100, atol=1e-3)
    np.testing.assert_allclose(after[:, 1], before[:, 1])


def test_apply_expression_canceled(window, monkeypatch):
    before = _xy(window.view)
    _apply(window, monkeypatch, "x += 100", ok=False)
    np.testing.assert_array_equal(_xy(window.view), before)


def test_apply_expression_to_all_channels(window2, monkeypatch):
    before = [_xy(window2.view, k) for k in range(2)]
    _apply(window2, monkeypatch, "y += 50", channel=APPLY_TO_ALL)
    for k in range(2):
        after = _xy(window2.view, k)
        np.testing.assert_allclose(
            after[:, 1], before[k][:, 1] + 50, atol=1e-3
        )


def test_apply_expression_to_the_second_channel(window2, monkeypatch):
    before = [_xy(window2.view, k) for k in range(2)]
    _apply(window2, monkeypatch, "x += 10", channel=window2.view.locs_paths[1])
    np.testing.assert_array_equal(_xy(window2.view, 0), before[0])
    assert not np.allclose(_xy(window2.view, 1), before[1])


def test_apply_flip_exchanges_x_and_y(window, monkeypatch):
    before = _xy(window.view)
    _apply(window, monkeypatch, "flip x y")
    np.testing.assert_allclose(_xy(window.view), before[:, ::-1])


@pytest.mark.parametrize("cmd", ["flip x z", "flip z x"])
def test_apply_flip_exchanges_x_and_z(cmd, window_3d, monkeypatch):
    view = window_3d.view
    x, z = view.locs[0]["x"].to_numpy(), view.locs[0]["z"].to_numpy()
    _, width = view.movie_size()  # nm
    _apply(window_3d, monkeypatch, cmd)
    # z is centered on the field of view
    np.testing.assert_allclose(view.locs[0]["x"], z + width / 2, atol=1e-2)
    np.testing.assert_allclose(view.locs[0]["z"], x, atol=1e-2)


def test_apply_spiral_and_undo(window, monkeypatch, answers):
    before = _xy(window.view)
    _apply(window, monkeypatch, "spiral 200 3")
    spiraled = _xy(window.view)
    assert not np.allclose(spiraled, before)
    assert np.abs(spiraled - before).max() <= 200 + 1e-3  # the radius
    _apply(window, monkeypatch, "uspiral")
    np.testing.assert_allclose(_xy(window.view), before, atol=1e-3)
    answers.record_messages()
    _apply(window, monkeypatch, "uspiral")  # undone already
    assert answers.messages == ["Localizations have not been spiraled yet."]
    np.testing.assert_allclose(_xy(window.view), before, atol=1e-3)


def test_apply_undo_spiral_needs_a_spiral(window, monkeypatch, answers):
    answers.record_messages()
    before = _xy(window.view)
    _apply(window, monkeypatch, "uspiral")
    assert answers.messages == ["Localizations have not been spiraled yet."]
    np.testing.assert_array_equal(_xy(window.view), before)


def test_apply_spiral_beyond_the_edge_and_undo(window, monkeypatch):
    """A spiral beyond the canvas grows it, which translates the
    localizations; undoing the spiral must restore them in the new
    frame."""
    from picasso import lib

    def camera_frame() -> np.ndarray:
        info = window.view.infos[0]
        offset = [
            lib.get_from_metadata(info, key, default=0.0) or 0.0
            for key in lib.CANVAS_OFFSET_KEYS
        ]
        return _xy(window.view) - np.array(offset)

    before = camera_frame()
    _apply(window, monkeypatch, "spiral 2000 1")
    assert window.view.locs[0]["x"].min() >= 0
    _apply(window, monkeypatch, "uspiral")
    np.testing.assert_allclose(camera_frame(), before, atol=1e-2)


# ---------------------------------------------------------------------------
# Reopening rotated localizations
# ---------------------------------------------------------------------------


def _open_rotated(qapp, path, monkeypatch) -> gui_render.Window:
    """A new Render window that opened ``path`` with Open rotated
    localizations."""
    from tests.test_gui_render_units import _wait

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *a, **k: ([str(path)], "*.hdf5"),
    )
    window = gui_render.Window(plugins_loaded=True)
    window.open_rotated_locs()
    _wait(qapp, lambda: window.view._load_thread is None)
    _wait(qapp, lambda: window.window_rot.isVisible())
    return window


def test_rotated_localizations_reopen_with_pick_and_rotation(
    window_3d, answers, monkeypatch, qt_offscreen, tmp_path
):
    from scipy.spatial.transform import Rotation

    view = window_3d.view
    window_3d.tools_settings_dialog.pick_shape.setCurrentText("Circle")
    window_3d.tools_settings_dialog.pick_diameter.setValue(1500.0)
    view._picks = [(2000.0, 2000.0)]
    window_3d.open_3d_view()
    view_rot = window_3d.window_rot.view_rot
    rotation = Rotation.from_rotvec([0.0, 0.5, 0.2])
    view_rot.set_rotation(rotation)
    out = tmp_path / "rotated.hdf5"
    answers.item(view.locs_paths[0])
    answers.save_path(out)
    window_3d.window_rot.save_locs_rotated()
    window_3d.window_rot.close()

    window = _open_rotated(qt_offscreen, out, monkeypatch)
    assert window.view._pick_shape == "Circle"
    assert window.view._picks == [[2000.0, 2000.0]]
    assert window.tools_settings_dialog.pick_shape.currentText() == "Circle"
    assert window.view._pick_size == 1500.0
    reopened = window.window_rot.view_rot.rotation
    assert (reopened * rotation.inv()).magnitude() < 1e-6
    window.window_rot.close()
    window.close()


def _rotated_file(tmp_path, entry: dict) -> str:
    """3D localizations saved with the rotation metadata ``entry``."""
    from picasso import io
    from tests.test_gui_render_slicer import _info as info_3d
    from tests.test_gui_render_slicer import _locs as locs_3d

    path = str(tmp_path / "rotated.hdf5")
    info = info_3d() + [{"angx": 0.1, "angy": 0.2, "angz": 0.3, **entry}]
    io.save_locs(path, locs_3d(z_range=200.0), info)
    return path


@pytest.mark.parametrize(
    "shape, pick, size, size_box",
    [
        ("Rectangle", [[500.0, 500.0], [3000.0, 3000.0]], 400.0, "pick_width"),
        ("Square", [2000.0, 2000.0], 800.0, "pick_side_length"),
        ("Polygon", [[0.0, 0.0], [3000, 0], [3000, 3000], [0, 0]], None, None),
    ],
)
def test_rotated_localizations_reopen_with_their_pick_shape(
    shape, pick, size, size_box, monkeypatch, qt_offscreen, tmp_path
):
    path = _rotated_file(
        tmp_path, {"Pick": pick, "Pick shape": shape, "Pick size (nm)": size}
    )
    window = _open_rotated(qt_offscreen, path, monkeypatch)
    assert window.view._pick_shape == shape
    assert window.view._picks == [pick]
    if size_box is not None:
        box = getattr(window.tools_settings_dialog, size_box)
        assert box.value() == size
    # legacy Euler angles
    from picasso import render

    saved = render.rotation_matrix(0.1, 0.2, 0.3)
    rotation = window.window_rot.view_rot.rotation
    assert (rotation * saved.inv()).magnitude() < 1e-6
    window.window_rot.close()
    window.close()


def test_rotated_brush_pick_reopens_with_its_strokes(
    monkeypatch, qt_offscreen, tmp_path
):
    strokes = [
        {"Width (nm)": 300.0, "Path": [[500.0, 500.0], [2500.0, 2500.0]]},
        {"Width (nm)": 200.0, "Path": [[500.0, 2500.0], [2500.0, 500.0]]},
    ]
    path = _rotated_file(
        tmp_path,
        {"Pick": strokes, "Pick shape": "Brush", "Pick size (nm)": None},
    )
    window = _open_rotated(qt_offscreen, path, monkeypatch)
    assert window.view._pick_shape == "Brush"
    assert window.view._picks == [
        [(300.0, strokes[0]["Path"]), (200.0, strokes[1]["Path"])]
    ]
    window.window_rot.close()
    window.close()


@pytest.mark.parametrize(
    "entry",
    [
        {
            "Pick": [[500.0, 1000.0], [2500.0, 3000.0]],
            "Pick shape": "Field of view",
        },
        {},  # saved before pick shapes were stored
    ],
)
def test_rotated_field_of_view_reopens_without_a_pick(
    entry, monkeypatch, qt_offscreen, tmp_path
):
    path = _rotated_file(tmp_path, entry)
    window = _open_rotated(qt_offscreen, path, monkeypatch)
    assert window.view._picks == []
    assert window.info_dialog.n_picks.text() == "0"
    if entry:  # the field of view saved
        (y0, x0), (y1, x1) = window.view.viewport
        assert x0 <= 500.0 and x1 >= 2500.0
        assert y0 <= 1000.0 and y1 >= 3000.0
    window.window_rot.close()
    window.close()


# ---------------------------------------------------------------------------
# G5M
# ---------------------------------------------------------------------------


def _group(window) -> None:
    """Label the localizations of every channel by their cluster, as
    DBSCAN would."""
    centers = np.array(_cluster_centers())
    for locs in window.view.locs:
        xy = locs[["x", "y"]].to_numpy()
        distances = np.linalg.norm(xy[:, None] - centers[None], axis=-1)
        locs["group"] = np.argmin(distances, axis=1).astype(np.int32)
    # as when clustered localizations are opened
    window.view.group_color = gui_render.render.get_group_color(
        window.view.locs[0]
    )


def _g5m_dialog(monkeypatch, ok=True, clustered=False, **setup) -> list:
    """Answer the G5M dialog; ``setup`` maps attributes of the dialog
    to a function called with them before closing it."""
    opened = []

    def exec_(dialog):
        opened.append(dialog)
        dialog.multiprocessing_check.setChecked(False)
        dialog.clustered_check.setChecked(clustered)
        for name, set_ in setup.items():
            set_(getattr(dialog, name))
        codes = QtWidgets.QDialog.DialogCode
        return codes.Accepted if ok else codes.Rejected

    monkeypatch.setattr(gui_render.G5MDialog, "exec", exec_)
    return opened


def test_g5m_needs_clustered_localizations(window, answers, monkeypatch):
    answers.record_messages()
    _g5m_dialog(monkeypatch)
    window.view.g5m()
    assert "does not contain the 'group' column" in answers.messages[0]


def test_g5m_canceled(window, answers, monkeypatch, tmp_path):
    _group(window)
    opened = _g5m_dialog(monkeypatch, ok=False)
    window.view.g5m()
    assert len(opened) == 1
    assert _saved(tmp_path) == []


def test_g5m_saves_the_molecules(
    window, answers, monkeypatch, qt_offscreen, tmp_path
):
    from picasso import io

    _group(window)
    _g5m_dialog(monkeypatch, clustered=True)
    answers.save_path(tmp_path / "mols.hdf5", tmp_path / "clusters.hdf5")
    window.view.g5m()
    _wait_for_tasks(qt_offscreen)
    molecules, _ = io.load_locs(str(tmp_path / "mols.hdf5"))
    assert len(molecules) > 0
    clustered, _ = io.load_locs(str(tmp_path / "clusters.hdf5"))
    assert len(clustered) > 0
    # the checks saved next to the molecules
    assert (tmp_path / "mols_subcluster_check.png").exists()
    assert (tmp_path / "mols_relsigma_check.png").exists()


@pytest.mark.parametrize("cancel", ["molecules", "clusters"])
def test_g5m_canceled_saving(
    cancel, window, answers, monkeypatch, qt_offscreen, tmp_path
):
    _group(window)
    _g5m_dialog(monkeypatch, clustered=True)
    if cancel == "molecules":
        answers.save_path("")
    else:
        answers.save_path(tmp_path / "mols.hdf5", "")
    window.view.g5m()
    _wait_for_tasks(qt_offscreen)
    assert _saved(tmp_path) == []


def test_g5m_of_all_channels(
    window2, answers, monkeypatch, qt_offscreen, tmp_path
):
    _group(window2)
    _g5m_dialog(monkeypatch, clustered=True)
    answers.item(APPLY_TO_ALL)
    answers.text("_mols", "_cl")
    window2.view.g5m()
    _wait_for_tasks(qt_offscreen)
    saved = _saved(tmp_path)
    for k in range(2):
        assert f"ch{k}_mols.hdf5" in saved
        assert f"ch{k}_cl.hdf5" in saved


@pytest.mark.parametrize("canceled", [["_mols", None], [None]])
def test_g5m_of_all_channels_canceled_suffix(
    canceled, window2, answers, monkeypatch, qt_offscreen, tmp_path
):
    _group(window2)
    _g5m_dialog(monkeypatch, clustered=True)
    answers.item(APPLY_TO_ALL)
    answers.text(*canceled)
    window2.view.g5m()
    _wait_for_tasks(qt_offscreen)
    assert _saved(tmp_path) == []


@pytest.mark.parametrize("remove", [True, False])
def test_g5m_asks_about_fiducial_clusters(
    remove, window, answers, monkeypatch
):
    _group(window)
    window.view.infos[0][0]["Frames"] = 100  # clusters of 150 locs
    asked = []
    button = QtWidgets.QMessageBox.StandardButton

    def question(*args, **kwargs):  # called on an instance
        asked.append(next(a for a in args if "fiducial" in str(a)))
        return button.Yes if remove else button.No

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", question)
    started = []
    monkeypatch.setattr(
        window.view,
        "_g5m_in_channel_job",
        lambda channel, params, *paths: started.append(params) or None,
    )
    _g5m_dialog(monkeypatch)
    answers.save_path("mols.hdf5")
    window.view.g5m()
    assert "contains clusters which can be fiducial markers" in asked[0]
    max_locs = started[0]["max_locs_per_cluster"][0]
    assert max_locs == (40 if remove else np.inf)


def test_g5m_dialog_sigma_bounds(window, answers, monkeypatch):
    _group(window)
    started = []
    monkeypatch.setattr(
        window.view,
        "_g5m_in_channel_job",
        lambda channel, params, *paths: started.append(params) or None,
    )
    _g5m_dialog(
        monkeypatch,
        loc_prec_handling=lambda combo: combo.setCurrentIndex(1),
    )
    answers.save_path("mols.hdf5")
    window.view.g5m()
    params = started[0]
    assert params["loc_prec_handle"] == "abs"
    assert params["sigma_bounds"] == (5.0, 20.0)  # nm
    assert params["group_column"] == "group"


def test_g5m_dialog_back_to_local_precision(window):
    dialog = gui_render.G5MDialog(window, 0)
    defaults = dialog.min_sigma.value(), dialog.max_sigma.value()
    dialog.loc_prec_handling.setCurrentIndex(1)
    assert dialog.min_sigma_label.text() == "Min. σ (nm):"
    dialog.loc_prec_handling.setCurrentIndex(0)
    assert dialog.min_sigma_label.text() == "Min. σ factor:"
    assert (dialog.min_sigma.value(), dialog.max_sigma.value()) == defaults
    dialog.close()


def test_g5m_dialog_offers_the_input_group_column(window):
    window.view.locs[0]["group_input"] = 0
    dialog = gui_render.G5MDialog(window, 0)
    items = [
        dialog.group_column.itemText(i)
        for i in range(dialog.group_column.count())
    ]
    assert items == ["group", "group_input"]
    dialog.close()


def _calibration(tmp_path, name="calib.yaml", magnification=None) -> str:
    import yaml

    calibration = {
        "X Coefficients": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "Y Coefficients": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
    if magnification is not None:
        calibration["Magnification factor"] = magnification
    path = tmp_path / name
    path.write_text(yaml.dump(calibration))
    return str(path)


def test_g5m_3d_canceled_without_a_calibration(window_3d, monkeypatch):
    """OK stays disabled until a 3D calibration is loaded, so Cancel is
    the only way out; reading the missing calibration then raised
    AttributeError."""
    opened = _g5m_dialog(monkeypatch, ok=False)
    params, ok = gui_render.G5MDialog.getParams(window_3d, 0)
    assert not opened[0].buttons.buttons()[0].isEnabled()
    assert not ok


@pytest.mark.parametrize("magnification", [0.8, None])
def test_g5m_3d_with_a_loaded_calibration(
    magnification, window_3d, answers, monkeypatch, tmp_path
):
    path = _calibration(tmp_path, magnification=magnification)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *a, **k: (path, "*.yaml"),
    )
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getDouble", lambda *a, **k: (0.75, True)
    )
    opened = _g5m_dialog(
        monkeypatch, load_calib_button=lambda button: button.click()
    )
    params, ok = gui_render.G5MDialog.getParams(window_3d, 0)
    assert ok
    assert opened[0].load_calib_button.text() == "3D calibration loaded"
    assert params["mode"] == "astigmatism"
    expected = 0.75 if magnification is None else magnification
    assert params["calibration"]["Magnification factor"] == expected


def test_g5m_3d_calibration_canceled_magnification(
    window_3d, monkeypatch, tmp_path
):
    window_3d.view.infos[0][0]["Z Calibration Path"] = _calibration(tmp_path)
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getDouble", lambda *a, **k: (0.0, False)
    )
    opened = _g5m_dialog(monkeypatch)
    params, ok = gui_render.G5MDialog.getParams(window_3d, 0)
    # loaded automatically from the metadata
    assert opened[0].calibration is not None
    assert (params, ok) == (None, False)


def test_g5m_3d_rejects_a_file_that_is_no_calibration(
    window_3d, monkeypatch, tmp_path
):
    import yaml

    errors = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        lambda parent, title, text, *a: errors.append(text),
    )
    path = tmp_path / "other.yaml"
    path.write_text(yaml.dump({"Pixelsize": 130}))
    dialog = gui_render.G5MDialog(window_3d, 0)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *a, **k: (str(path), "*.yaml"),
    )
    dialog.load_calib_button.click()
    assert errors[0].startswith("other.yaml could not be opened")
    assert dialog.calibration is None
    assert not dialog.buttons.buttons()[0].isEnabled()
    dialog.close()


@pytest.mark.parametrize(
    "table, expected",
    [
        ({"Area (LP^2)": [10.0, 1000.0, 20.0, 5000.0]}, "50.0%"),
        ({"other": [1.0]}, None),
    ],
)
def test_g5m_check_cluster_sizes(
    table, expected, window, answers, monkeypatch, tmp_path
):
    import pandas as pd

    answers.record_messages()
    path = tmp_path / "areas.csv"
    pd.DataFrame(table).to_csv(path, index=False)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *a, **k: (str(path), "*.csv"),
    )
    dialog = gui_render.G5MDialog(window, 0)
    dialog.cluster_size_button.click()
    if expected is None:
        assert answers.messages == [
            "Could not read the cluster areas/volumes file."
        ]
    else:
        assert dialog.cluster_size_label.text().endswith(expected)
    dialog.close()


def test_g5m_check_cluster_sizes_needs_one_channel(window2, answers):
    answers.record_messages()
    dialog = gui_render.G5MDialog(window2, 2)  # all channels
    dialog.cluster_size_button.click()
    assert answers.messages == [
        "Please select a single channel to check cluster sizes."
    ]
    dialog.close()


# ---------------------------------------------------------------------------
# Parameter dialogs, filled in by the user
# ---------------------------------------------------------------------------


def _fill(monkeypatch, dialog_class, ok=True, **values) -> list:
    """Answer ``dialog_class`` by setting its widgets to ``values``
    (spin boxes, check boxes and combo boxes); returns the dialogs
    shown."""
    shown = []

    def exec_(dialog):
        shown.append(dialog)
        for name, value in values.items():
            widget = getattr(dialog, name)
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(value)
            elif isinstance(widget, QtWidgets.QComboBox):
                widget.setCurrentText(value)
            else:
                widget.setValue(value)
        codes = QtWidgets.QDialog.DialogCode
        return codes.Accepted if ok else codes.Rejected

    monkeypatch.setattr(dialog_class, "exec", exec_)
    return shown


@pytest.mark.parametrize(
    "method, dialog_class, values, suffix",
    [
        (
            "dbscan",
            gui_render.DbscanDialog,
            dict(radius=30.0, density=4, min_locs=10, save_centers=True),
            "_dbscan",
        ),
        (
            "hdbscan",
            gui_render.HdbscanDialog,
            dict(min_cluster=50, min_samples=10, save_centers=True),
            "_hdbscan",
        ),
        (
            "smlm_clusterer",
            gui_render.SMLMDialog,
            dict(radius_xy=39.0, min_locs=10, save_centers=True),
            "_clustered",
        ),
    ],
)
def test_clustering_with_the_parameters_entered(
    method,
    dialog_class,
    values,
    suffix,
    window,
    answers,
    monkeypatch,
    qt_offscreen,
    tmp_path,
):
    from picasso import io
    from tests.test_gui_render_tasks import N_CLUSTERS

    shown = _fill(monkeypatch, dialog_class, **values)
    out = tmp_path / f"out{suffix}.hdf5"
    answers.save_path(out)
    getattr(window.view, method)()
    _wait_for_tasks(qt_offscreen)
    assert len(shown) == 1
    clustered, _ = io.load_locs(str(out))
    assert clustered["group"].nunique() == N_CLUSTERS
    assert (tmp_path / f"out{suffix}_centers.hdf5").exists()


@pytest.mark.parametrize(
    "dialog_class", [gui_render.DbscanDialog, gui_render.SMLMDialog]
)
@pytest.mark.parametrize("flag_3d", [False, True])
def test_clustering_dialog_asks_for_the_z_radius_in_3d(
    dialog_class, flag_3d, window
):
    dialog = dialog_class(window, flag_3D=flag_3d)
    z_radius = dialog.radius_z
    assert (z_radius.parent() is dialog) is flag_3d
    dialog.close()


def test_export_view_with_the_parameters_entered(
    window, answers, monkeypatch, tmp_path
):
    from PIL import Image

    window.display_settings_dlg.dynamic_disp_px.setChecked(False)
    (y_min, x_min), (y_max, x_max) = window.view.viewport
    shown = _fill(
        monkeypatch,
        gui_render.ExportKwargsDialog,
        width=2000.0,
        height=1000.0,
        disp_px_size=10.0,
        blur_method="None",
    )
    answers.save_path(tmp_path / "v.png")
    window.export_kwargs()
    dialog = shown[0]
    # filled in with the current view
    assert dialog.x.value() == pytest.approx(x_min, abs=0.01)
    assert dialog.y.value() == pytest.approx(y_min, abs=0.01)
    with Image.open(tmp_path / "v.png") as image:
        assert image.size == (200, 100)


def test_export_view_canceled(window, answers, monkeypatch, tmp_path):
    _fill(monkeypatch, gui_render.ExportKwargsDialog, ok=False)
    window.export_kwargs()
    assert _saved(tmp_path) == []


def test_link_with_the_parameters_entered(window, monkeypatch, qt_offscreen):
    shown = _fill(
        monkeypatch, gui_render.LinkDialog, max_distance=20.0, max_dark_time=1
    )
    n_locs = len(window.view.locs[0])
    window.view.link()
    _wait_for_tasks(qt_offscreen)
    assert shown[0].max_dark_time.value() == 1
    assert len(window.view.locs[0]) < n_locs
    assert window.view.infos[0][-1]["Max. distance (nm)"] == 20.0
    assert window.view.infos[0][-1]["Max. transient dark frames"] == 1


def test_link_canceled(window, monkeypatch, qt_offscreen):
    _fill(monkeypatch, gui_render.LinkDialog, ok=False)
    n_locs = len(window.view.locs[0])
    window.view.link()
    _wait_for_tasks(qt_offscreen)
    assert len(window.view.locs[0]) == n_locs


def test_aim_with_the_parameters_entered(window, monkeypatch, qt_offscreen):
    shown = _fill(
        monkeypatch,
        gui_render.AIMDialog,
        segmentation=500,
        intersect_d=20.0,
        max_drift=60.0,
    )
    window.view.undrift_aim()
    _wait_for_tasks(qt_offscreen)
    assert len(shown) == 1
    assert window.view._drift[0] is not None


# ---------------------------------------------------------------------------
# Saving localizations
# ---------------------------------------------------------------------------

COMBINE = "Combine all channels"


def _load(path):
    from picasso import io

    return io.load_locs(str(path))


def test_save_localizations_of_a_channel(window, answers, tmp_path):
    out = tmp_path / "saved.hdf5"
    answers.save_path(out)
    window.save_locs()
    locs, info = _load(out)
    assert len(locs) == len(window.view.locs[0])
    assert "Last driftfile" in info[-1]


def test_save_localizations_of_all_channels(window2, answers, tmp_path):
    answers.item(APPLY_TO_ALL)
    answers.text("_mine")
    window2.save_locs()
    for k in range(2):
        locs, _ = _load(tmp_path / f"ch{k}_mine.hdf5")
        assert len(locs) == len(window2.view.locs[k])


def test_save_localizations_combined(window2, answers, tmp_path):
    out = tmp_path / "both.hdf5"
    answers.item(COMBINE)
    answers.save_path(out)
    window2.save_locs()
    locs, info = _load(out)
    assert len(locs) == sum(len(locs) for locs in window2.view.locs)
    assert np.all(np.diff(locs["frame"]) >= 0)
    assert info[-1]["Paths to combined files"] == window2.view.locs_paths


@pytest.mark.parametrize("cancel", ["channel", "path", "suffix"])
def test_save_localizations_canceled(cancel, window2, answers, tmp_path):
    if cancel == "channel":
        answers.item(None)
    elif cancel == "path":
        answers.item(window2.view.locs_paths[0])
        answers.save_path("")
    else:
        answers.item(APPLY_TO_ALL)
        answers.text(None)
    window2.save_locs()
    assert _saved(tmp_path) == []


def test_save_localizations_needs_localizations(qt_offscreen, answers):
    answers.record_messages()
    window = gui_render.Window(plugins_loaded=True)
    for save in (
        window.save_locs,
        window.save_picked_locs,
        window.save_picked_locs_separately,
    ):
        save()
    assert len(answers.messages) == 3
    window.close()


def test_save_picked_localizations(window, answers, tmp_path):
    _circle_picks(window, n=2)
    out = tmp_path / "picked.hdf5"
    answers.save_path(out)
    window.save_picked_locs()
    locs, info = _load(out)
    assert sorted(set(locs["group"])) == [0, 1]
    assert info[-1]["Pick Shape"] == "Circle"


def test_save_picked_localizations_of_all_channels(window2, answers, tmp_path):
    _circle_picks(window2, n=2)
    answers.item(APPLY_TO_ALL)
    answers.text("_p")
    window2.save_picked_locs()
    for k in range(2):
        locs, _ = _load(tmp_path / f"ch{k}_p.hdf5")
        n_picked = sum(len(p) for p in window2.view.picked_locs(k))
        assert len(locs) == n_picked
    assert n_picked == 0  # the picks are around the clusters of ch0


def test_save_picked_localizations_combined(window2, answers, tmp_path):
    _circle_picks(window2, n=2)
    out = tmp_path / "both.hdf5"
    answers.item(COMBINE)
    answers.save_path(out)
    window2.save_picked_locs()
    locs, info = _load(out)
    singles = [
        sum(len(p) for p in window2.view.picked_locs(k)) for k in range(2)
    ]
    assert len(locs) == sum(singles)


@pytest.mark.parametrize(
    "method", ["save_picked_locs", "save_picked_locs_separately"]
)
def test_save_picked_localizations_without_picks(
    method, window2, answers, tmp_path
):
    """Saving picked localizations without picks raised TypeError."""
    answers.record_messages()
    answers.item(COMBINE)
    answers.save_path(tmp_path / "picked.hdf5")
    getattr(window2, method)()
    assert answers.messages == ["No localizations picked. Please pick first."]
    assert _saved(tmp_path) == []


def test_save_each_pick_separately(window, answers, tmp_path):
    _circle_picks(window, n=2)
    answers.save_path(tmp_path / "pick.hdf5")
    window.save_picked_locs_separately()
    for i in range(2):
        locs, info = _load(tmp_path / f"pick_{i}.hdf5")
        assert len(locs) > 0
        assert info[-1]["Area (um^2)"] > 0


def test_save_each_pick_separately_of_all_channels(window2, answers, tmp_path):
    _circle_picks(window2, n=2)
    answers.item(APPLY_TO_ALL)
    answers.text("_pick")
    window2.save_picked_locs_separately()
    assert _saved(tmp_path) == [
        f"ch{k}_pick_{i}.{ext}"
        for k in range(2)
        for i in range(2)
        for ext in ("hdf5", "yaml")
    ]


def test_save_each_pick_separately_combined(window2, answers, tmp_path):
    _circle_picks(window2, n=2)
    answers.item(COMBINE)
    answers.save_path(tmp_path / "both.hdf5")
    window2.save_picked_locs_separately()
    for i in range(2):
        locs, info = _load(tmp_path / f"both_{i}.hdf5")
        n_locs = sum(len(window2.view.picked_locs(k)[i]) for k in range(2))
        assert len(locs) == n_locs
        assert info[-1]["Channels combined"] == window2.view.locs_paths


@pytest.mark.parametrize("proceed", [False, True])
def test_save_many_picks_separately_asks_first(
    proceed, window, answers, monkeypatch, tmp_path
):
    _circle_picks(window, n=12)
    asked = []
    button = QtWidgets.QMessageBox.StandardButton
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *a, **k: asked.append(a[2])
        or (button.Yes if proceed else button.No),
    )
    answers.save_path(tmp_path / "pick.hdf5")
    window.save_picked_locs_separately()
    assert asked[0].startswith("12 picks were selected.")
    assert len(_saved(tmp_path)) == (24 if proceed else 0)


# ---------------------------------------------------------------------------
# Exporting images
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scalebar", [False, True])
def test_export_current_view(scalebar, window, answers, tmp_path):
    from picasso import io

    window.display_settings_dlg.scalebar_groupbox.setChecked(scalebar)
    answers.save_path(tmp_path / "view.png")
    window.export_current()
    expected = ["view.png", "view.yaml"]
    if not scalebar:  # a copy with a scale bar, too
        expected.append("view_scalebar.png")
    assert _saved(tmp_path) == sorted(expected)
    # the scale bar is hidden again
    checked = window.display_settings_dlg.scalebar_groupbox.isChecked()
    assert checked is scalebar
    info = io.load_info(str(tmp_path / "view.yaml"))[0]
    assert info["Localizations loaded"] == window.view.locs_paths


@pytest.mark.parametrize("ext", [".pdf", ".svg", ".tif"])
def test_export_current_view_formats(ext, window, answers, tmp_path):
    window.display_settings_dlg.scalebar_groupbox.setChecked(True)
    answers.save_path(tmp_path / f"view{ext}")
    answers.integer(150)  # DPI of a PDF
    window.export_current()
    assert _saved(tmp_path) == sorted([f"view{ext}", "view.yaml"])


def test_export_current_view_canceled(window, answers, tmp_path):
    answers.save_path(tmp_path / "view.pdf", "")
    answers.integer(None)  # no DPI
    window.export_current()
    window.export_current()
    assert _saved(tmp_path) == []


def test_export_current_view_is_named_after_the_first_shown_channel(
    window2, answers, monkeypatch
):
    suggested = []

    def ask(parent, title, path, *args, **kwargs):
        suggested.append(path)
        return "", ""

    monkeypatch.setattr(gui_render.lib, "get_save_filename_ext_dialog", ask)
    window2.dataset_dialog.checks[0].setChecked(False)
    window2.export_current()
    assert suggested[0].endswith("ch1_view.png")


def test_export_current_view_with_all_channels_hidden(
    window2, answers, monkeypatch
):
    """With no channel shown, finding the first shown one raised
    ValueError."""
    suggested = []

    def ask(parent, title, path, *args, **kwargs):
        suggested.append(path)
        return "", ""

    monkeypatch.setattr(gui_render.lib, "get_save_filename_ext_dialog", ask)
    for check in window2.dataset_dialog.checks:
        check.setChecked(False)
    window2.export_current()
    assert suggested[0].endswith("ch0_view.png")


def test_export_current_view_with_the_property_colorbar(
    window, answers, tmp_path
):
    from picasso import io

    settings = window.display_settings_dlg
    settings.parameter.setCurrentText("photons")
    settings.render_check.setChecked(True)
    answers.save_path(tmp_path / "view.png")
    window.export_current()
    colorbar = "view" + gui_render.COLORBAR_SUFFIX
    assert colorbar + io.colorbar_export_format() in _saved(tmp_path)
    info = io.load_info(str(tmp_path / "view.yaml"))[0]
    assert info["Render property"] == "photons"


@pytest.mark.parametrize("ext", [".png", ".pdf"])
def test_export_complete_image(ext, window, answers, tmp_path):
    from PIL import Image

    window.view.zoom_in()  # the complete image is exported nonetheless
    answers.save_path(tmp_path / f"all{ext}")
    answers.integer(100)
    window.export_complete()
    assert _saved(tmp_path) == sorted([f"all{ext}", "all.yaml"])
    if ext == ".png":
        with Image.open(tmp_path / "all.png") as image:
            width, height = image.size
        assert width == pytest.approx(height, abs=1)


def test_export_complete_image_canceled(window, answers, tmp_path):
    answers.save_path(tmp_path / "all.pdf", "")
    answers.integer(None)
    window.export_complete()
    window.export_complete()
    assert _saved(tmp_path) == []


# ---------------------------------------------------------------------------
# CBC and spatial statistics, with the dialogs filled in
# ---------------------------------------------------------------------------


def _linked(window) -> None:
    """Mark every channel as linked, so that no warning is shown."""
    for k, locs in enumerate(window.view.locs):
        window.view.locs[k] = locs.assign(len=np.int32(1))


def _accept_with(monkeypatch, dialog_class, setup) -> list:
    """Answer ``dialog_class`` by calling ``setup(dialog)`` and then its
    own ``accept`` (which may refuse); returns the dialogs shown."""
    shown = []

    def exec_(dialog):
        shown.append(dialog)
        setup(dialog)
        dialog.accept()
        return dialog.result()

    monkeypatch.setattr(dialog_class, "exec", exec_)
    return shown


def test_cbc_needs_two_channels(window, answers):
    answers.record_messages()
    window.view.cbc()
    assert "requires at least two channels" in answers.messages[0]


def test_cbc_of_one_pair_with_the_dialog(
    window2, answers, monkeypatch, qt_offscreen, tmp_path
):
    _linked(window2)

    def setup(dialog):
        dialog.names[1].setText("B")
        dialog.scored.setCurrentIndex(0)
        dialog.partner.setCurrentIndex(1)
        dialog.r_max.setValue(100.0)
        dialog.steps.setValue(10)

    _accept_with(monkeypatch, gui_render.CBCDialog, setup)
    out = tmp_path / "scored.hdf5"
    answers.save_path(out)
    window2.view.cbc()
    _wait_for_tasks(qt_offscreen)
    locs, info = _load(out)
    assert "cbc_B" in locs.columns
    assert info[-1]["Step"] == "CBC"
    assert info[-1]["CBC partner names"] == ["B"]


def test_cbc_refuses_the_same_channel_twice(window2, answers, monkeypatch):
    answers.record_messages()

    def setup(dialog):
        dialog.partner.setCurrentIndex(0)

    shown = _accept_with(monkeypatch, gui_render.CBCDialog, setup)
    answers.save_path("")
    window2.view.cbc()
    assert answers.messages == ["Please choose two different channels."]
    assert shown[0].result() != QtWidgets.QDialog.DialogCode.Accepted


def test_cbc_refuses_a_name_without_letters(window2, answers, monkeypatch):
    answers.record_messages()

    def setup(dialog):
        dialog.names[1].setText("--")

    _accept_with(monkeypatch, gui_render.CBCDialog, setup)
    window2.view.cbc()
    assert "containing letters or digits" in answers.messages[0]


@pytest.mark.parametrize("overwrite", [False, True])
def test_cbc_of_all_pairs_asks_before_overwriting(
    overwrite, window2, monkeypatch, qt_offscreen, tmp_path
):
    _linked(window2)
    _accept_with(
        monkeypatch,
        gui_render.CBCDialog,
        lambda dialog: (
            dialog.all_pairs.setChecked(True),
            dialog.steps.setValue(10),
        ),
    )
    (tmp_path / "out").mkdir()
    existing = tmp_path / "out" / "ch0_cbc.hdf5"
    existing.write_text("old")
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        lambda *a, **k: str(tmp_path / "out"),
    )
    asked = []
    button = QtWidgets.QMessageBox.StandardButton
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *a, **k: asked.append(a[2])
        or (button.Yes if overwrite else button.No),
    )
    window2.view.cbc()
    _wait_for_tasks(qt_offscreen)
    assert "ch0_cbc.hdf5" in asked[0]
    assert (existing.read_bytes() != b"old") is overwrite
    assert (tmp_path / "out" / "ch1_cbc.hdf5").exists() is overwrite


@pytest.mark.parametrize("cancel", ["dialog", "folder"])
def test_cbc_canceled(cancel, window2, monkeypatch, qt_offscreen, tmp_path):
    _linked(window2)
    if cancel == "dialog":
        _fill(monkeypatch, gui_render.CBCDialog, ok=False)
    else:
        _accept_with(
            monkeypatch,
            gui_render.CBCDialog,
            lambda dialog: dialog.all_pairs.setChecked(True),
        )
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getExistingDirectory", lambda *a, **k: ""
    )
    window2.view.cbc()
    _wait_for_tasks(qt_offscreen)
    assert _saved(tmp_path) == []


def _molecules(window) -> None:
    """Mark every channel as molecules, so that no warning is shown."""
    for k, locs in enumerate(window.view.locs):
        window.view.locs[k] = locs.assign(n_events=np.int32(1))


@pytest.mark.parametrize(
    "statistic, setup",
    [
        ("ripley_h", lambda d: d.region.setCurrentIndex(1)),  # the FOV
        ("rdf", lambda d: d.partner.setCurrentIndex(2)),  # cross
        ("nnd", lambda d: None),
    ],
)
def test_spatial_statistic_with_the_dialog(
    statistic, setup, window2, monkeypatch, qt_offscreen
):
    _molecules(window2)

    def fill(dialog):
        dialog.simulations.setValue(1)
        dialog.steps.setValue(5)
        dialog.r_max.setValue(100.0)
        setup(dialog)

    _accept_with(monkeypatch, gui_render.SpatialStatisticDialog, fill)
    window2.view.spatial_statistic(statistic)
    _wait_for_tasks(qt_offscreen)
    plot_window = window2.view.spatial_statistic_windows[statistic]
    assert plot_window.result is not None
    plot_window.close()


def test_spatial_statistic_canceled(window, monkeypatch, qt_offscreen):
    _fill(monkeypatch, gui_render.SpatialStatisticDialog, ok=False)
    window.view.spatial_statistic("ripley_h")
    _wait_for_tasks(qt_offscreen)
    assert "ripley_h" not in window.view.spatial_statistic_windows


@pytest.mark.parametrize(
    "statistic, setup, message",
    [
        (
            "nnd",
            lambda d: [
                d.channels_list.item(i).setCheckState(
                    gui_render.QtCore.Qt.CheckState.Unchecked
                )
                for i in range(2)
            ],
            "channel",
        ),
        ("ripley_h", lambda d: d.partner.setCurrentIndex(1), "channel"),
    ],
)
def test_spatial_statistic_dialog_refusals(
    statistic, setup, message, window2, answers, monkeypatch
):
    answers.record_messages()
    shown = _accept_with(monkeypatch, gui_render.SpatialStatisticDialog, setup)
    window2.view.spatial_statistic(statistic)
    assert len(answers.messages) == 1 and message in answers.messages[0]
    assert shown[0].result() != QtWidgets.QDialog.DialogCode.Accepted


def test_spatial_statistic_dialog_in_3d(window_3d, answers, monkeypatch):
    dialog = gui_render.SpatialStatisticDialog(window_3d, "ripley_h")
    assert dialog.use_z.isEnabled()
    z = window_3d.view.locs[0]["z"]
    assert dialog.z_min.value() == np.floor(z.min())
    assert dialog.z_max.value() == np.ceil(z.max())
    dialog.use_z.setChecked(True)
    assert not dialog.z_limits.isHidden()
    assert dialog.z_min.isHidden()
    dialog.z_limits.setCurrentIndex(1)  # manual
    assert not dialog.z_min.isHidden()
    assert not dialog.simulated_z.isHidden()
    dialog.partner.setCurrentIndex(2)
    assert dialog.simulated_z.isHidden()
    assert not dialog.cross_null.isHidden()
    # an empty z range is refused
    answers.record_messages()
    dialog.z_min.setValue(100.0)
    dialog.z_max.setValue(100.0)
    dialog.accept()
    assert "z" in answers.messages[0]
    dialog.close()

    def setup(dialog):
        dialog.use_z.setChecked(True)
        dialog.z_limits.setCurrentIndex(1)
        dialog.z_min.setValue(-50.0)
        dialog.z_max.setValue(50.0)
        dialog.simulated_z.setCurrentIndex(1)

    _accept_with(monkeypatch, gui_render.SpatialStatisticDialog, setup)
    params, ok = gui_render.SpatialStatisticDialog.getParams(
        window_3d, "ripley_h"
    )
    assert ok
    assert params["z_range"] == (-50.0, 50.0)
    assert params["simulated_z"] == "observed"


# ---------------------------------------------------------------------------
# Files dialog: titles and colors of the channels
# ---------------------------------------------------------------------------


def test_channel_title_is_changed_and_reset(window, answers):
    dataset = window.dataset_dialog
    original = dataset.checks[0].text()
    answers.text("Bassoon", None, "reset")
    name = dataset.title[0].objectName()
    dataset.change_title(name)
    assert dataset.checks[0].text() == "Bassoon"
    dataset.change_title(name)  # canceled
    assert dataset.checks[0].text() == "Bassoon"
    dataset.change_title(name)
    assert dataset.checks[0].text() == original


def test_channel_title_of_a_short_path_is_reset(
    qt_offscreen, answers, tmp_path
):
    """A path of up to 40 characters is the title itself; resetting it
    showed "reset"."""
    from tests.test_gui_render_tasks import _info, _locs

    window = _window(tmp_path)
    window.view.add("short.hdf5", _locs(), _info(), render_=False)
    dataset = window.dataset_dialog
    assert dataset.checks[1].text() == "short.hdf5"
    answers.text("Bassoon", "Reset")
    name = dataset.title[1].objectName()
    dataset.change_title(name)
    dataset.change_title(name)
    assert dataset.checks[1].text() == "short.hdf5"
    window.close()


def test_channel_color_from_the_color_picker(window, answers, monkeypatch):
    from PyQt6 import QtGui

    answers.record_messages()
    dataset = window.dataset_dialog
    colors = [QtGui.QColor(255, 128, 0), QtGui.QColor()]  # 2nd: canceled
    monkeypatch.setattr(
        QtWidgets.QColorDialog, "getColor", lambda *a: colors.pop(0)
    )
    name = dataset.title[0].objectName()
    dataset.auto_colors.setChecked(True)
    dataset.select_channel_color(name)
    assert answers.messages == [
        "Untick 'Automatic coloring' to set colors manually."
    ]
    dataset.auto_colors.setChecked(False)
    dataset.select_channel_color(name)
    assert dataset.colorselection[0].currentText() == "#FF8000"
    brightest = dataset._channel_luts[0][-1]
    assert brightest == pytest.approx((1.0, 128 / 255, 0.0), abs=1e-3)
    dataset.select_channel_color(name)  # canceled: unchanged
    assert dataset.colorselection[0].currentText() == "#FF8000"
    dataset.select_channel_color("no such channel")


# ---------------------------------------------------------------------------
# Unfolding groups, fiducials and rendering by property
# ---------------------------------------------------------------------------


def test_unfold_picks_onto_a_grid(window, answers):
    _circle_picks(window, n=4)
    answers.integer(2, 1000)  # 2 per column, 1000 nm apart
    window.view.unfold_groups_square()
    locs = window.view.locs[0]
    assert "group" not in locs.columns
    assert window.view._picks == []
    # the picks were moved apart onto a 2 x 2 grid
    assert np.ptp(locs["x"]) > 1000
    assert len(locs) < 4 * 200  # only the picked localizations


@pytest.mark.parametrize("answer", [[None], [2, None]])
def test_unfold_picks_canceled_keeps_the_localizations(
    answer, window, answers
):
    """Canceling kept only the picked localizations."""
    _circle_picks(window, n=4)
    before = window.view.locs[0].copy()
    answers.integer(*answer)
    window.view.unfold_groups_square()
    locs = window.view.locs[0]
    assert len(locs) == len(before)
    assert list(locs.columns) == list(before.columns)
    assert len(window.view._picks) == 4


def test_unfold_clusters_onto_a_grid(window, answers):
    from tests.test_gui_render_tasks import N_CLUSTERS

    _group(window)
    n_locs = len(window.view.locs[0])
    answers.integer(5, 500)
    window.view.unfold_groups_square()
    locs = window.view.locs[0]
    assert len(locs) == n_locs
    assert locs["group"].nunique() == N_CLUSTERS


def test_unfold_needs_groups_and_one_channel(window, window2, answers):
    answers.record_messages()
    window.view.unfold_groups_square()
    window2.view.unfold_groups_square()
    assert "need to be grouped" in answers.messages[0]
    assert answers.messages[1] == "Please load only one channel."


def _window_with_a_fiducial(tmp_path) -> gui_render.Window:
    import pandas as pd

    from tests.test_gui_render_tasks import (
        HEIGHT,
        N_FRAMES,
        PIXELSIZE,
        WIDTH,
        _info,
        _locs,
    )

    locs = _locs()
    rng = np.random.default_rng(3)
    fiducial = locs.iloc[:N_FRAMES].copy()
    fiducial["frame"] = np.arange(N_FRAMES, dtype=np.uint32)
    fiducial["x"] = np.float32(4000.0) + rng.normal(0, 5, N_FRAMES)
    fiducial["y"] = np.float32(3000.0) + rng.normal(0, 5, N_FRAMES)
    locs = pd.concat([locs, fiducial], ignore_index=True)
    locs = locs.sort_values("frame", kind="stable").reset_index(drop=True)
    window = gui_render.Window(plugins_loaded=True)
    window.view.add(str(tmp_path / "fid.hdf5"), locs, _info(), render_=False)
    window.view.viewport = [
        (0.0, 0.0),
        (HEIGHT * PIXELSIZE, WIDTH * PIXELSIZE),
    ]
    return window


def test_pick_fiducials(qt_offscreen, tmp_path):
    window = _window_with_a_fiducial(tmp_path)
    window.view._pick_shape = "Circle"
    window.view.pick_fiducials()
    ((x, y),) = window.view._picks
    assert (x, y) == pytest.approx((4000.0, 3000.0), abs=100.0)
    assert window.view._pick_size > 500.0  # nm
    window.close()


def test_pick_fiducials_refusals(window, answers):
    answers.record_messages()
    window.view._pick_shape = "Square"
    window.view.pick_fiducials()
    window.view._pick_shape = "Circle"
    window.view._picks = [(100.0, 100.0)]
    window.view.pick_fiducials()
    window.view._picks = []
    window.view.pick_fiducials()  # no fiducial in the clusters
    assert answers.messages == [
        "Please select circular pick before picking fiducials.",
        "Please remove all picks before picking fiducials.",
        "No fiducials found, manual picking is required.",
    ]


def test_render_by_property(window):
    settings = window.display_settings_dlg
    settings.parameter.setCurrentText("photons")
    window.view.set_property()  # as when chosen
    photons = window.view.locs[0]["photons"]
    assert settings.maximum_render.value() == pytest.approx(
        photons.max(), rel=1e-3
    )
    assert settings.minimum_render.value() == pytest.approx(
        photons.min(), rel=1e-3
    )
    assert settings.render_check.isEnabled()
    assert not settings.render_check.isChecked()
    settings.render_check.setChecked(True)
    assert window.view.x_render_state


def _three_locs_per_channel(tmp_path) -> gui_render.Window:
    """Two channels of 3 localizations each, around (1000, 1000) nm."""
    import pandas as pd

    from tests.test_gui_render_tasks import HEIGHT, PIXELSIZE, WIDTH, _info

    window = gui_render.Window(plugins_loaded=True)
    for k in range(2):
        locs = pd.DataFrame(
            {
                "frame": np.arange(3, dtype=np.uint32),
                "x": np.float32([990.0, 1000.0, 1010.0]) + k,
                "y": np.float32([1000.0, 1010.0, 990.0]),
                "z": np.float32([-10.0, 0.0, 10.0]),
                "photons": np.float32([1000.0] * 3),
                "lpx": np.float32([5.0] * 3),
                "lpy": np.float32([5.0] * 3),
            }
        )
        window.view.add(
            str(tmp_path / f"tiny{k}.hdf5"), locs, _info(), render_=False
        )
    window.view.viewport = [
        (0.0, 0.0),
        (HEIGHT * PIXELSIZE, WIDTH * PIXELSIZE),
    ]
    window.view._pick_shape = "Circle"
    window.tools_settings_dialog.pick_diameter.setValue(200.0)
    window.view._picks = [(1000.0, 1000.0)]
    return window


@pytest.mark.parametrize(
    "method, dialog_class",
    [
        ("show_pick", None),
        ("show_pick_3d", gui_render.PlotDialog),
        ("show_pick_3d_iso", gui_render.PlotDialogIso),
    ],
)
def test_scatter_of_all_channels_has_the_channel_colors(
    method, dialog_class, qt_offscreen, answers, monkeypatch, tmp_path
):
    """The color of a channel was taken for values to color by when a
    pick held as many localizations as the color has components."""
    from matplotlib import colors as mcolors

    window = _three_locs_per_channel(tmp_path)
    answers.item("Show all channels")
    collections = []
    if dialog_class is None:
        real_close = plt.close

        def close(fig=None):
            collections.extend(fig.axes[0].collections)
            real_close(fig)

        monkeypatch.setattr(gui_render.plt, "close", close)
        _pick_replies(monkeypatch, "Accept")
    else:
        shown = _plot_dialog_replies(monkeypatch, dialog_class, "yes")
    getattr(window.view, method)()
    if dialog_class is not None:
        collections = [c for ax in shown[0][2] for c in ax.collections]
    expected = [mcolors.to_rgba(c) for c in gui_render.lib.get_colors(2)]
    for k, collection in enumerate(collections):
        face = tuple(collection.get_facecolor()[0])
        assert face == pytest.approx(expected[k % 2], abs=1e-6)
    window.close()


# ---------------------------------------------------------------------------
# Mask image
# ---------------------------------------------------------------------------


@pytest.fixture
def mask_dialog(window, answers):
    """The mask dialog, opened from the menu, of the clustered
    localizations at 260 nm pixels with a narrow blur."""
    answers.record_messages()
    dialog = window.mask_settings_dialog
    dialog.init_dialog()
    dialog.disp_px_size.setValue(260.0)
    dialog.mask_blur.setValue(100)
    yield dialog
    dialog.close()


def test_mask_dialog_shows_the_steps(mask_dialog):
    titles = [plot._title for plot in mask_dialog.plots]
    assert titles == ["Histogrammed localizations", "Blur", "Mask"]
    assert mask_dialog.mask.shape == mask_dialog.H.shape
    assert mask_dialog.save_mask_button.isEnabled()


@pytest.mark.parametrize("method", ["Otsu", "Li", "Mean", "Triangle"])
def test_mask_threshold_methods(method, mask_dialog):
    mask_dialog.thresh_method.setCurrentText(method)
    mask_dialog.update_plots()
    assert not mask_dialog.mask_thresh.isEnabled()
    assert 0 < mask_dialog.mask.sum() < mask_dialog.mask.size
    mask_dialog.thresh_method.setCurrentText("Custom")
    mask_dialog.update_plots()
    assert mask_dialog.mask_thresh.isEnabled()


def test_mask_histogram(mask_dialog):
    mask_dialog.show_hist()
    window = mask_dialog.hist_window
    assert window.figure.axes[0].get_title() == (
        "Density of blurred image values"
    )
    window.redraw()
    window.close()


def test_masked_localizations_are_saved(
    mask_dialog, window, answers, tmp_path
):
    from picasso import lib

    mask_dialog.save_locs()  # nothing masked yet
    assert len(answers.messages) == 1
    mask_dialog.mask_locs()
    assert mask_dialog.plots[3]._title == "Masked"
    answers.save_path(tmp_path / "in.hdf5", tmp_path / "out.hdf5")
    mask_dialog.save_locs()
    locs_in, info_in = _load(tmp_path / "in.hdf5")
    locs_out, info_out = _load(tmp_path / "out.hdf5")
    assert len(locs_in) + len(locs_out) == len(window.view.locs[0])
    assert len(locs_in) > len(locs_out)  # the clusters are in the mask
    # the mask spans the field of view
    width, height = lib.field_of_view(window.view.infos[0])
    assert info_in[-1]["Area (um^2)"] + info_out[-1]["Area (um^2)"] == (
        pytest.approx(width * height * 1e-6, rel=1e-6)
    )
    # the metadata masking.mask_locs returned
    assert info_in[-1]["Step"] == "Mask in"
    assert info_out[-1]["Step"] == "Mask out"
    assert info_in[-1]["Display pixel size (nm)"] == 260.0
    assert info_in[-1]["Blur (nm)"] == 100


@pytest.mark.parametrize("cancel", [1, 2])
def test_masked_localizations_canceled(cancel, mask_dialog, answers, tmp_path):
    mask_dialog.mask_locs()
    paths = [tmp_path / "in.hdf5", tmp_path / "out.hdf5"]
    paths[cancel - 1] = ""
    answers.save_path(*paths)
    mask_dialog.save_locs()
    assert _saved(tmp_path) == []


def test_mask_of_the_second_channel(window2, answers, tmp_path):
    """Only the chosen channel is masked, at index 0 of the masked
    localizations; saving read the index of the channel instead."""
    dialog = window2.mask_settings_dialog
    answers.item(window2.view.locs_paths[1])
    dialog.init_dialog()
    dialog.mask_locs()
    answers.save_path(tmp_path / "in.hdf5", tmp_path / "out.hdf5")
    dialog.save_locs()
    locs_in, _ = _load(tmp_path / "in.hdf5")
    locs_out, _ = _load(tmp_path / "out.hdf5")
    assert len(locs_in) + len(locs_out) == len(window2.view.locs[1])
    dialog.close()


def test_mask_canceled_channel(window2, answers):
    """Canceling the channel selection raised TypeError."""
    dialog = window2.mask_settings_dialog
    answers.item(None)
    dialog.init_dialog()
    assert not dialog.isVisible()


def test_mask_all_channels(window2, answers, tmp_path):
    dialog = window2.mask_settings_dialog
    answers.item(window2.view.locs_paths[0])
    dialog.init_dialog()
    dialog.save_all.setChecked(True)
    dialog.mask_locs()
    assert len(dialog.index_locs) == 2
    answers.text("_in", "_out")
    dialog.save_locs()
    assert _saved(tmp_path) == sorted(
        f"ch{k}_{side}.{ext}"
        for k in range(2)
        for side in ("in", "out")
        for ext in ("hdf5", "yaml")
    )
    answers.text("_in", None)  # canceled
    dialog.save_locs()
    dialog.close()


def test_mask_all_channels_beyond_the_mask(window2, answers):
    """A channel with localizations beyond the field of view of the
    channel the mask was made from: they are outside the mask (were
    indexed past the mask)."""
    import pandas as pd

    from picasso import lib

    width, height = lib.field_of_view(window2.view.infos[0])
    far = window2.view.locs[1].iloc[:1].copy()
    far["x"] = np.float32(1.5 * width)
    window2.view.locs[1] = pd.concat(
        [window2.view.locs[1], far], ignore_index=True
    )
    info = [dict(window2.view.infos[1][0])]
    info[0]["Width"] *= 2
    window2.view.infos[1] = info + window2.view.infos[1][1:]
    dialog = window2.mask_settings_dialog
    answers.item(window2.view.locs_paths[0])
    dialog.init_dialog()
    dialog.save_all.setChecked(True)
    dialog.mask_locs()
    assert len(dialog.index_locs) == 2
    assert 1.5 * width in dialog.index_locs_out[1]["x"].tolist()
    assert dialog.index_infos[1][-1]["Step"] == "Mask in"
    dialog.close()


def test_mask_and_blur_saved_and_mask_loaded(
    mask_dialog, answers, monkeypatch, tmp_path
):
    answers.save_path(tmp_path / "mask.npy", tmp_path / "blur.png")
    mask_dialog.save_mask()
    mask_dialog.save_blur()
    assert _saved(tmp_path) == ["blur.png", "mask.npy", "mask.png"]
    saved = np.load(tmp_path / "mask.npy")
    np.testing.assert_array_equal(saved, mask_dialog.mask)

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *a, **k: (str(tmp_path / "mask.npy"), "*.npy"),
    )
    mask_dialog.mask = None
    mask_dialog.save_button.setEnabled(False)
    mask_dialog.load_mask()
    np.testing.assert_array_equal(mask_dialog.mask, saved)
    assert mask_dialog.save_button.isEnabled()


def test_mask_dialog_suggests_the_last_directory(
    mask_dialog, window, tmp_path
):
    window.pwd = str(tmp_path / "ch0.hdf5")  # a file: its folder
    assert mask_dialog.last_used_directory() == str(tmp_path)
    window.pwd = str(tmp_path)
    assert mask_dialog.last_used_directory() == str(tmp_path)
    window.pwd = []
    assert mask_dialog.last_used_directory() == str(tmp_path)


# ---------------------------------------------------------------------------
# Dropping files onto the window
# ---------------------------------------------------------------------------


def _drop(window, *paths) -> None:
    from PyQt6 import QtCore, QtGui

    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(path)) for path in paths])
    event = QtGui.QDropEvent(
        QtCore.QPointF(10.0, 10.0),
        QtCore.Qt.DropAction.CopyAction,
        mime,
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    window.view.dropEvent(event)


def test_drag_enter_accepts_files_only(window):
    from PyQt6 import QtCore, QtGui

    for urls, accepted in [([QtCore.QUrl.fromLocalFile("a.hdf5")], True)]:
        mime = QtCore.QMimeData()
        mime.setUrls(urls)
        event = QtGui.QDragEnterEvent(
            QtCore.QPoint(1, 1),
            QtCore.Qt.DropAction.CopyAction,
            mime,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        window.view.dragEnterEvent(event)
        assert event.isAccepted() is accepted
    mime = QtCore.QMimeData()
    mime.setText("text")
    event = QtGui.QDragEnterEvent(
        QtCore.QPoint(1, 1),
        QtCore.Qt.DropAction.CopyAction,
        mime,
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    window.view.dragEnterEvent(event)
    assert not event.isAccepted()


def test_dropped_localizations_are_loaded(window, qt_offscreen, tmp_path):
    from picasso import io
    from tests.test_gui_render_tasks import _info, _locs
    from tests.test_gui_render_units import _wait

    path = tmp_path / "dropped.hdf5"
    io.save_locs(str(path), _locs(seed=5), _info())
    _drop(window, path, tmp_path / "ignored.npy")
    _wait(qt_offscreen, lambda: window.view._load_thread is None)
    assert window.view.locs_paths[-1] == str(path)


def test_dropped_view_settings_are_applied(window, answers, tmp_path):
    settings = window.display_settings_dlg
    settings.dynamic_disp_px.setChecked(False)
    settings.maximum.setValue(7.0)
    window.view.zoom_in()
    viewport = np.array(window.view.viewport, dtype=float)
    answers.save_path(tmp_path / "view.png")
    window.export_current()
    settings.maximum.setValue(3.0)
    window.view.fit_in_view()
    _drop(window, tmp_path / "view.yaml")
    assert settings.maximum.value() == 7.0
    np.testing.assert_allclose(window.view.viewport, viewport, atol=1.0)


def test_dropped_view_settings_with_unknown_colors(window, answers, tmp_path):
    from picasso import io

    answers.record_messages()
    io.save_info(
        str(tmp_path / "view.yaml"),
        [
            {
                io.FORMAT_VERSION_ATTR: io.FORMAT_VERSION,
                gui_render.FOV_KEY: [0.0, 0.0, 4000.0, 4000.0],
                "Max. density": 5.0,
                "Colors": ["no such color"],
                "Scalebar length (nm)": 300,  # before Picasso 1.0
                "Min. blur (nm)": 2.0,
                "Blur method": "One-pixel",
            }
        ],
    )
    _drop(window, tmp_path / "view.yaml")
    assert "no such color" in answers.messages[0]
    assert window.display_settings_dlg.scalebar.value() == 300.0
    assert window.display_settings_dlg.min_blur_width.value() == 2.0


def test_dropped_picks_are_loaded(window, tmp_path):
    picks = _circle_picks(window, n=2)
    window.view.save_picks(str(tmp_path / "picks.yaml"))
    window.view._picks = []
    _drop(window, tmp_path / "picks.yaml")
    assert np.allclose(window.view._picks, picks)


def test_dropped_yaml_of_another_kind_is_ignored(window, tmp_path):
    (tmp_path / "list.yaml").write_text("- 1\n- 2\n")
    (tmp_path / "other.yaml").write_text("Key: 1\n")
    _drop(window, tmp_path / "list.yaml")
    _drop(window, tmp_path / "other.yaml")
    assert window.view._picks == []


def test_dropped_image_is_overlaid(window, tmp_path):
    from PIL import Image

    path = tmp_path / "image.png"
    Image.fromarray(np.full((16, 16), 128, dtype=np.uint8)).save(path)
    _drop(window, path)
    overlay = window.image_overlay_dialog
    assert overlay.path == str(path)
    overlay.close()


def test_dropped_fov_is_shown(window, tmp_path):
    window.info_dialog.change_fov.x_box.setValue(1000.0)
    window.info_dialog.change_fov.y_box.setValue(500.0)
    window.info_dialog.change_fov.w_box.setValue(2000.0)
    window.info_dialog.change_fov.h_box.setValue(2000.0)
    path = tmp_path / "fov.txt"
    gui_render.save_fov_file(str(path), np.array([1000, 500, 2000, 2000]))
    _drop(window, path)
    (y_min, x_min), (y_max, x_max) = window.view.viewport
    assert x_min <= 1000.0 + 1 and x_max >= 3000.0 - 1
    assert y_min <= 500.0 + 1 and y_max >= 2500.0 - 1


def test_fov_saved_and_loaded_in_the_info_dialog(
    window, answers, monkeypatch, tmp_path
):
    change_fov = window.info_dialog.change_fov
    change_fov.x_box.setValue(100.0)
    change_fov.w_box.setValue(1500.0)
    change_fov.h_box.setValue(1500.0)
    answers.save_path(tmp_path / "fov.txt", "")
    change_fov.save_fov()
    change_fov.save_fov()  # canceled
    assert _saved(tmp_path) == ["fov.txt"]
    change_fov.x_box.setValue(0.0)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *a, **k: (str(tmp_path / "fov.txt"), "*.txt"),
    )
    change_fov.load_fov()
    assert change_fov.x_box.value() == 100.0
    (_, x_min), _ = window.view.viewport
    assert x_min == pytest.approx(100.0, abs=1.0)


def _drift_file(path, n_frames, n_dims=2, shift=10.0) -> None:
    import pandas as pd

    from picasso import io

    columns = ["x", "y", "z"][:n_dims]
    drift = pd.DataFrame({c: np.full(n_frames, shift) for c in columns})
    io.save_drift(str(path), drift)


def test_dropped_drift_is_applied(window, tmp_path):
    from tests.test_gui_render_tasks import N_FRAMES

    before = _xy(window.view)
    _drift_file(tmp_path / "drift.txt", N_FRAMES)
    _drop(window, tmp_path / "drift.txt")
    np.testing.assert_allclose(_xy(window.view), before - 10.0, atol=1e-3)


def test_dropped_drift_of_all_channels(window2, answers, tmp_path):
    from tests.test_gui_render_tasks import N_FRAMES

    before = [_xy(window2.view, k) for k in range(2)]
    answers.item(APPLY_TO_ALL)
    _drift_file(tmp_path / "drift.txt", N_FRAMES)
    _drop(window2, tmp_path / "drift.txt")
    for k in range(2):
        np.testing.assert_allclose(
            _xy(window2.view, k), before[k] - 10.0, atol=1e-3
        )


@pytest.mark.parametrize("all_channels", [False, True])
def test_dropped_drift_of_another_movie_is_refused(
    all_channels, window2, answers, tmp_path
):
    answers.record_messages()
    answers.item(APPLY_TO_ALL if all_channels else window2.view.locs_paths[0])
    before = _xy(window2.view)
    _drift_file(tmp_path / "drift.txt", 10)
    _drop(window2, tmp_path / "drift.txt")
    assert "frames" in answers.messages[0]
    np.testing.assert_array_equal(_xy(window2.view), before)


def test_dropped_text_that_is_no_drift_or_fov(window, tmp_path):
    (tmp_path / "words.txt").write_text("not numbers\n")
    (tmp_path / "three.txt").write_text("1 2 3\n")
    before = _xy(window.view)
    _drop(window, tmp_path / "words.txt")
    _drop(window, tmp_path / "three.txt")
    np.testing.assert_array_equal(_xy(window.view), before)


# ---------------------------------------------------------------------------
# Filtering picks, binding event cores and drift
# ---------------------------------------------------------------------------


def test_filter_picks_by_count(window, answers):
    from tests.test_gui_render_tasks import LOCS_PER_CLUSTER

    picks = _circle_picks(window, n=3)
    window.view._picks.append((10.0, 10.0))  # empty, in a corner
    figures = _figures()
    answers.integer(10, LOCS_PER_CLUSTER)  # min., max.
    window.view.filter_picks()
    assert window.view._picks == picks
    assert window.info_dialog.n_picks.text() == "3"
    # the histogram is shown in the view, not in a pyplot window
    assert _figures() == figures


def test_filter_picks_by_count_of_all_channels(window2, answers):
    picks = _circle_picks(window2, n=3)
    answers.item(APPLY_TO_ALL)
    answers.integer(10, 10_000)
    window2.view.filter_picks()
    # the picks are around the clusters of the first channel only
    assert window2.view._picks == []
    del picks


def test_filter_picks_canceled(window, answers):
    picks = _circle_picks(window, n=3)
    answers.integer(None)
    window.view.filter_picks()
    answers.integer(1, None)
    window.view.filter_picks()
    assert window.view._picks == picks


def test_filter_square_picks_by_count(window, answers):
    window.tools_settings_dialog.pick_shape.setCurrentText("Square")
    window.tools_settings_dialog.pick_side_length.setValue(300.0)
    window.view._picks = list(_cluster_centers()[:2]) + [(10.0, 10.0)]
    answers.integer(1, 10_000)
    window.view.filter_picks()
    assert len(window.view._picks) == 2


def test_binding_event_cores(window, monkeypatch):
    shown = _fill(monkeypatch, gui_render.LinkDialog, max_distance=50.0)
    info = window.view.infos[0]
    n_locs = len(window.view.locs[0])
    window.view.select_binding_event_cores()
    assert shown[0].max_dark_time.value() == 0  # suggested
    locs = window.view.locs[0]
    assert 0 < len(locs) < n_locs
    assert "group" in locs.columns
    entry = window.view.infos[0][-1]
    assert entry["Max. distance (nm)"] == 50.0
    # a new list: the old one may be shared with a linked window
    assert window.view.infos[0] is not info
    assert len(info) == 1


def test_binding_event_cores_refusals(window, answers, monkeypatch):
    answers.record_messages()
    # no two localizations of a cluster in consecutive frames this close
    _fill(monkeypatch, gui_render.LinkDialog, max_distance=0.001)
    locs = window.view.locs[0]
    window.view.select_binding_event_cores()
    assert "No localizations left" in answers.messages[0]
    assert window.view.locs[0] is locs
    _fill(monkeypatch, gui_render.LinkDialog, ok=False)
    window.view.select_binding_event_cores()
    window.view.locs[0] = locs.assign(len=np.int32(1))
    window.view.select_binding_event_cores()
    assert answers.messages[1] == "Localizations are already linked. Aborting."
    assert len(answers.messages) == 2


def test_drift_adds_up_and_is_saved(window, tmp_path):
    import pandas as pd

    from picasso import io

    view = window.view
    n = 5
    view.add_drift(0, pd.DataFrame({"x": np.ones(n), "y": np.ones(n)}))
    view.add_drift(
        0,
        pd.DataFrame(
            {"x": np.ones(n), "y": np.full(n, 2.0), "z": np.full(n, 3.0)}
        ),
    )
    drift = view._drift[0]
    np.testing.assert_array_equal(drift["x"], 2.0)
    np.testing.assert_array_equal(drift["y"], 3.0)
    np.testing.assert_array_equal(drift["z"], 3.0)
    saved = io.load_drift(view._driftfiles[0])
    np.testing.assert_array_equal(saved["z"], 3.0)
    view.add_drift(
        0,
        pd.DataFrame({"x": np.zeros(n), "y": np.zeros(n), "z": np.ones(n)}),
    )
    np.testing.assert_array_equal(view._drift[0]["z"], 4.0)


def _mouse(x, y, button=None, modifiers=None):
    """The parts of a Qt mouse or wheel event the mask plots read."""
    from types import SimpleNamespace

    from PyQt6 import QtCore

    return SimpleNamespace(
        position=lambda: QtCore.QPointF(x, y),
        angleDelta=lambda: QtCore.QPoint(0, 120),
        button=lambda: button or QtCore.Qt.MouseButton.RightButton,
        modifiers=lambda: modifiers or QtCore.Qt.KeyboardModifier.NoModifier,
    )


def test_mask_plots_zoom_and_pan_together(mask_dialog, monkeypatch):
    from PyQt6 import QtCore

    first, second = mask_dialog.plots[:2]
    first.wheelEvent(_mouse(150, 150))  # without Ctrl: scrolls
    assert first._zoom == 1.0
    monkeypatch.setattr(
        QtWidgets.QApplication,
        "keyboardModifiers",
        staticmethod(lambda: QtCore.Qt.KeyboardModifier.ControlModifier),
    )
    first.wheelEvent(_mouse(250, 250))  # toward the bottom right
    assert first._zoom == pytest.approx(1.15)
    assert first._center_x > 0.5 and first._center_y > 0.5
    assert second._zoom == first._zoom  # linked
    center = first._center_x
    first.mousePressEvent(_mouse(100, 100))
    first.mouseMoveEvent(_mouse(150, 100))  # drag to the right
    first.mouseReleaseEvent(_mouse(150, 100))
    assert first._center_x < center
    assert second._center_x == first._center_x
    first.mouseMoveEvent(_mouse(10, 10))  # no longer dragging
    assert second._center_x == first._center_x
    # Ctrl + left button pans, too
    first.mousePressEvent(
        _mouse(
            100,
            100,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
    )
    assert first._drag_start is not None
    first.mouseReleaseEvent(_mouse(100, 100, QtCore.Qt.MouseButton.LeftButton))
    first.mouseDoubleClickEvent(_mouse(0, 0))
    assert (second._zoom, second._center_x) == (1.0, 0.5)
    # a newly shown plot takes the zoom of its siblings
    first.wheelEvent(_mouse(150, 150))
    mask_dialog.mask_locs()
    assert mask_dialog.plots[3]._zoom == first._zoom


# ---------------------------------------------------------------------------
# Nearest neighbors, pick properties and metadata
# ---------------------------------------------------------------------------


def test_nearest_neighbor_distances_between_channels(
    window2, answers, tmp_path
):
    answers.item(*window2.view.locs_paths)  # from channel 0 to channel 1
    answers.integer(2)
    out = tmp_path / "nn.hdf5"
    answers.save_path(out)
    window2.view.nearest_neighbor()
    locs, info = _load(out)
    assert len(locs) == len(window2.view.locs[0])
    assert (locs["nnd_1"] <= locs["nnd_2"]).all()
    assert info[-1]["Channel 2"] == window2.view.locs_paths[1]


@pytest.mark.parametrize(
    "items, integers, paths",
    [([None], [], []), (["0", None], [], []), (["0", "1"], [None], [])],
)
def test_nearest_neighbor_canceled(
    items, integers, paths, window2, answers, tmp_path
):
    names = {"0": window2.view.locs_paths[0], "1": window2.view.locs_paths[1]}
    answers.item(*[names.get(item) for item in items])
    answers.integer(*integers)
    answers.save_path(*paths)
    window2.view.nearest_neighbor()
    assert _saved(tmp_path) == []


def test_nearest_neighbor_of_3d_and_2d_channels(
    qt_offscreen, answers, tmp_path
):
    """A 3D channel and a 2D one were compared in different dimensions,
    which raised ValueError."""
    from tests.test_gui_render_tasks import _info, _locs

    window = _window_3d(tmp_path)
    window.view.add(
        str(tmp_path / "flat.hdf5"), _locs(), _info(), render_=False
    )
    answers.item(window.view.locs_paths[0], window.view.locs_paths[2])
    answers.integer(2)
    answers.save_path(tmp_path / "nn.hdf5")
    window.view.nearest_neighbor()
    locs, _ = _load(tmp_path / "nn.hdf5")
    assert np.isfinite(locs["nnd_2"]).all()
    window.close()


def test_nearest_neighbor_in_another_channel(window2, answers, tmp_path):
    answers.item(*window2.view.locs_paths)
    answers.integer(1)
    answers.save_path(tmp_path / "nn.hdf5")
    window2.view.nearest_neighbor()
    locs, _ = _load(tmp_path / "nn.hdf5")
    assert "nnd_1" in locs.columns


def test_pick_properties_of_a_channel(window, answers, qt_offscreen, tmp_path):
    import pandas as pd

    _circle_picks(window, n=3)
    out = tmp_path / "props.hdf5"
    answers.save_path(out)
    window.save_pick_properties()
    _wait_for_tasks(qt_offscreen)
    groups = pd.read_hdf(out, "groups")
    assert len(groups) == 3


def test_pick_properties_of_all_channels(
    window2, answers, qt_offscreen, tmp_path
):
    _group(window2)
    answers.item(APPLY_TO_ALL)
    answers.text("_props")
    window2.save_pick_properties()
    _wait_for_tasks(qt_offscreen)
    for k in range(2):
        assert (tmp_path / f"ch{k}_props.hdf5").exists()


def test_pick_properties_need_picks_or_groups(window, answers, qt_offscreen):
    answers.record_messages()
    answers.save_path("props.hdf5")
    window.save_pick_properties()
    _wait_for_tasks(qt_offscreen)
    assert answers.messages[0].startswith("No picks found.")


def test_metadata_dialog(window, answers, qt_offscreen):
    window.show_metadata()
    assert window.metadata_dialog.isVisible()
    window.metadata_dialog.close()
    answers.record_messages()
    empty = gui_render.Window(plugins_loaded=True)
    empty.show_metadata()
    assert answers.messages == ["No files loaded."]
    empty.close()


# ---------------------------------------------------------------------------
# Files dialog: background and channel order
# ---------------------------------------------------------------------------


def test_background_color_from_the_color_picker(window2, monkeypatch):
    from PyQt6 import QtGui

    colors = [QtGui.QColor(255, 255, 255), QtGui.QColor()]
    monkeypatch.setattr(
        QtWidgets.QColorDialog, "getColor", lambda *a: colors.pop(0)
    )
    dataset = window2.dataset_dialog
    dataset.select_background_color()
    assert dataset.background_color == (1.0, 1.0, 1.0)
    dataset.select_background_color()  # canceled
    assert dataset.background_color == (1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    "row, text, order",
    [
        (0, "Move down", ["ch1", "ch0"]),
        (1, "Move to top", ["ch1", "ch0"]),
        (0, "Move up", ["ch0", "ch1"]),  # disabled
    ],
)
def test_channel_order_from_the_row_menu(
    row, text, order, window2, monkeypatch
):
    from PyQt6 import QtCore

    shown = []

    def exec_(menu, *args):
        actions = {action.text(): action for action in menu.actions()}
        shown.append(sorted(t for t, a in actions.items() if a.isEnabled()))
        if actions[text].isEnabled():
            actions[text].trigger()

    monkeypatch.setattr(QtWidgets.QMenu, "exec", exec_)
    dataset = window2.dataset_dialog
    header = dataset.table.verticalHeader()
    y = header.sectionViewportPosition(row) + 2
    dataset._show_row_header_menu(QtCore.QPoint(2, y))
    names = [
        gui_render.os.path.splitext(gui_render.os.path.basename(p))[0]
        for p in window2.view.locs_paths
    ]
    assert names == order
    assert len(shown) == 1


def test_row_menu_needs_two_channels(window, monkeypatch):
    from PyQt6 import QtCore

    shown = []
    monkeypatch.setattr(
        QtWidgets.QMenu, "exec", lambda menu, *a: shown.append(menu)
    )
    window.dataset_dialog._show_row_header_menu(QtCore.QPoint(2, 2))
    assert shown == []


def test_unticked_channel_leaves_the_3d_window(window_3d):
    view = window_3d.view
    view._picks = []
    window_3d.open_3d_view()
    view_rot = window_3d.window_rot.view_rot
    view_rot.update_scene()
    window_3d.dataset_dialog.checks[1].setChecked(False)
    window_3d.dataset_dialog.update_rotation_window()
    assert window_3d.window_rot.isVisible()
    window_3d.window_rot.close()


# ---------------------------------------------------------------------------
# Display and tools leftovers
# ---------------------------------------------------------------------------


def _open_file(monkeypatch, path) -> None:
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *a, **k: (str(path), ""),
    )


def test_custom_colormap_from_a_file(window, monkeypatch, tmp_path):
    from matplotlib import colormaps

    path = tmp_path / "cmap.npy"
    np.save(path, colormaps["viridis"](np.linspace(0, 1, 256)))
    _open_file(monkeypatch, path)
    window.display_settings_dlg.colormap.setCurrentText("Custom")
    assert window.view.custom_cmap.shape == (256, 4)
    assert window.display_settings_dlg.colormap.currentText() == "Custom"


@pytest.mark.parametrize(
    "content, message",
    [
        (np.zeros((10, 4)), "Colormap must be of shape (256, 4)"),
        (np.full((256, 4), 2.0), "All elements of the colormap"),
        (b"not numpy", "Could not load"),
        (None, None),  # canceled
    ],
)
def test_custom_colormap_that_cannot_be_used(
    content, message, window, answers, monkeypatch, tmp_path
):
    """An unusable file raised ValueError instead of telling why."""
    answers.record_messages()
    path = tmp_path / "cmap.npy"
    if isinstance(content, bytes):
        path.write_bytes(content)
    elif content is not None:
        np.save(path, content)
    _open_file(monkeypatch, path if content is not None else "")
    window.display_settings_dlg.colormap.setCurrentText("Custom")
    assert window.display_settings_dlg.colormap.currentText() == "magma"
    if message is None:
        assert answers.messages == []
    else:
        assert answers.messages[0].startswith(message)


def test_move_to_pick(window, monkeypatch):
    picks = _circle_picks(window, n=3)
    offered = []

    def get_int(parent, title, label, value, minimum, maximum, *a):
        offered.append((minimum, maximum))
        return 2, True

    monkeypatch.setattr(QtWidgets.QInputDialog, "getInt", get_int)
    window.view.move_to_pick()
    # only existing picks can be chosen
    assert offered == [(0, 2)]
    (y_min, x_min), (y_max, x_max) = window.view.viewport
    x, y = picks[2]
    assert x_min < x < x_max and y_min < y < y_max
    assert x_max - x_min < 1000.0  # around the 300 nm pick


def test_remove_columns_from_the_dialog(window, monkeypatch):
    def exec_(dialog):
        assert not dialog.checks["x"].isEnabled()  # required
        dialog.checks["ellipticity"].setChecked(True)
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(gui_render.lib.RemoveColumnsDialog, "exec", exec_)
    window.remove_columns()
    assert "ellipticity" not in window.view.locs[0].columns
    assert window.view.infos[0][-1]["Removed columns"] == ["ellipticity"]


def test_remove_no_columns(window, monkeypatch):
    monkeypatch.setattr(
        gui_render.lib.RemoveColumnsDialog,
        "exec",
        lambda dialog: QtWidgets.QDialog.DialogCode.Accepted,
    )
    columns = list(window.view.locs[0].columns)
    window.remove_columns()
    assert list(window.view.locs[0].columns) == columns
    assert len(window.view.infos[0]) == 1


def test_channel_colors_are_saved(window2, answers, tmp_path):
    dataset = window2.dataset_dialog
    dataset.auto_colors.setChecked(False)
    dataset.colorselection[1].setCurrentText("cyan")
    answers.save_path(tmp_path / "colors.txt")
    dataset.save_colors()
    lines = (tmp_path / "colors.txt").read_text().splitlines()
    assert len(lines) == 2 and lines[1] == "cyan"


def test_pick_statistics(window, qt_offscreen):
    _circle_picks(window, n=3)
    window.view.update_pick_info_long()
    _wait_for_tasks(qt_offscreen)
    info_dialog = window.info_dialog
    assert float(info_dialog.n_localizations_mean.text()) > 100
    assert "dark" in info_dialog.pick_info
    info_dialog.pick_hist_window.close()


def test_test_clusterer_g5m_calibration(window, monkeypatch, tmp_path):
    import yaml

    params = window.test_clusterer_dialog.test_g5m_params
    errors = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        lambda parent, title, text, *a: errors.append(text),
    )
    other = tmp_path / "other.yaml"
    other.write_text(yaml.dump({"Pixelsize": 130}))
    _open_file(monkeypatch, other)
    params.load_calibration()
    assert params.calibration is None
    assert errors[0].startswith("other.yaml could not be opened")
    calibration = tmp_path / "calib.yaml"
    calibration.write_text(
        yaml.dump(
            {
                "picasso_format": "1.0",
                "X Coefficients": [0.0] * 6 + [1.0],
                "Y Coefficients": [0.0] * 6 + [1.0],
            }
        )
    )
    _open_file(monkeypatch, calibration)
    params.load_calibration()
    assert params.calibration["X Coefficients"][-1] == 1.0
    _open_file(monkeypatch, "")
    params.load_calibration()  # canceled: kept
    assert params.calibration is not None
