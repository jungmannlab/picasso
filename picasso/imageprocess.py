"""
    picasso/imageprocess
    ~~~~~~~~~~~~~~~~~~~~

    Image processing functions

    :author: Joerg Schnitzbauer, 2016
"""
import matplotlib.pyplot as _plt
from matplotlib.widgets import RectangleSelector as _RectangleSelector


_plt.style.use('ggplot')


def split(movie, info, frame=0, max=0.9, rectangle=None):
    if rectangle:
        pass
    else:
        subregions = []
        shifts = []
        infos = []

        def on_split_select(press_event, release_event):
            x1, y1 = press_event.xdata, press_event.ydata
            x2, y2 = release_event.xdata, release_event.ydata
            xmin = int(min(x1, x2) + 0.5)
            xmax = int(max(x1, x2) + 0.5)
            ymin = int(min(y1, y2) + 0.5)
            ymax = int(max(y1, y2) + 0.5)
            subregions.append(movie[:, ymin:ymax+1, xmin:xmax+1])
            shifts.append((ymin, xmin))
            subregion_info = info[:]
            subregion_info[0]['Height'] = ymax - ymin + 1
            subregion_info[0]['Width'] = xmax - xmin + 1
            infos.append(subregion_info)

        f = _plt.figure(figsize=(12, 12))
        ax = f.add_subplot(111)
        ax.matshow(movie[frame], cmap='viridis', vmax=max*movie[frame].max())
        ax.grid(False)
        selector = _RectangleSelector(ax, on_split_select, useblit=True, rectprops=dict(edgecolor='red', fill=False))
        _plt.show()
        return subregions, shifts, infos
