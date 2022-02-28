import time
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from numpy.lib.recfunctions import stack_arrays
from ... import postprocess, lib, io

class Plugin():
	def __init__(self, window):
		self.name = "render"
		self.window = window

	def execute(self):
		tools_menu = self.window.menus[2]
		tools_menu.addSeparator()
		self.window.fret_traces_action = tools_menu.addAction(
			"Show FRET traces"
		)
		self.window.fret_traces_action.triggered.connect(self.show_fret)

		self.window.calculate_fret_action = tools_menu.addAction(
			"Calculate FRET in picks"
		)
		self.window.calculate_fret_action.triggered.connect(
			self.calculate_fret_dialog
		)

	def show_fret(self):
		channel_acceptor = self.window.view.get_channel(
			title="Select acceptor channel"
		)
		channel_donor = self.window.view.get_channel(
			title="Select donor channel"
		)

		removelist = []

		n_channels = len(self.window.view.locs_paths)
		acc_picks = self.window.view.picked_locs(channel_acceptor)
		don_picks = self.window.view.picked_locs(channel_donor)

		if not self.window.view._picks:
			raise ValueError(
				"No picks found.  Please pick first."
			)
		else:
			if self.window.view._pick_shape == "Rectangle":
				raise NotImplementedError(
					"Not implemented for rectangle picks"
				)
			params = {}
			params["t0"] = time.time()
			i = 0
			while i < len(self.window.view._picks):
				pick = self.window.view._picks[i]

				fret_dict, fret_locs = postprocess.calculate_fret(
					acc_picks[i],
					don_picks[i],
				)

				fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
				fig.canvas.set_window_title("FRET-trace")
				ax1.plot(fret_dict["frames"], fret_dict["acc_trace"])
				ax1.set_title("Acceptor intensity vs frame")
				ax1.set_xlim(0, (fret_dict["maxframes"] + 1))
				ax1.set_ylabel("Photons")

				ax2.plot(fret_dict["frames"], fret_dict["don_trace"])
				ax2.set_title("Donor intensity vs frame")
				ax2.set_ylabel("Photons")

				ax3.scatter(
					fret_dict["fret_timepoints"], 
					fret_dict["fret_events"],
					s=2
				)
				ax3.set_title(r"$\frac{I_A}{I_D+I_A}$")
				ax3.set_xlabel("Frame")
				ax3.set_ylabel("Ratio")

				fig.canvas.draw()
				width, height = fig.canvas.get_width_height()

				im = QtGui.QImage(
					fig.canvas.buffer_rgba(),
					width,
					height,
					QtGui.QImage.Format_ARGB32,
				)

				self.window.view.setPixmap((QtGui.QPixmap(im)))
				self.window.view.setAlignment(QtCore.Qt.AlignCenter)

				params["n_removed"] = len(removelist)
				params["n_kept"] = i - params["n_removed"]
				params["n_total"] = len(self.window.view._picks)
				params["i"] = i

				msgBox = self.window.view.pick_message_box(params)

				reply = msgBox.exec()

				if reply == 0:
					# Accepted
					if pick in removelist:
						removelist.remove(pick)
				elif reply == 3:
					# Cancel
					break
				elif reply == 2:
					# Back
					if i >= 2:
						i -= 2
					else:
						i = -1
				else:
					# Discard
					removelist.append(pick)

				i += 1
				plt.close()

		for pick in removelist:
			self.window.view._picks.remove(pick)

		self.window.view.n_picks = len(self.window.view._picks)

		self.window.view.update_pick_info_short()
		self.window.view.update_scene()

	def calculate_fret_dialog(self):
		if self.window.view._pick_shape == "Rectangle":
			raise NotImplementedError(
				"Not implemented for rectangle picks"
			)
		print("Calculating FRET...")
		fret_events = []

		channel_acceptor = self.window.view.get_channel(
			title="Select acceptor channel"
		)
		channel_donor = self.window.view.get_channel(
			title="Select donor channel"
		)

		acc_picks = self.window.view.picked_locs(channel_acceptor)
		don_picks = self.window.view.picked_locs(channel_donor)

		K = len(self.window.view._picks)
		progress = lib.ProgressDialog(
			"Calculating fret in picks...",
			0,
			K,
			self.window.view,
		)
		progress.show()

		all_fret_locs = []

		for i in range(K):
			fret_dict, fret_locs = postprocess.calculate_fret(
				acc_picks[i],
				don_picks[i],
			)
			if fret_dict["fret_events"] != []:
				fret_events.append(fret_dict["fret_events"])
			if fret_locs != []:
				all_fret_locs.append(fret_locs)
			progress.set_value(i + 1)
		progress.close()

		if fret_events == []:
			raise ValueError(
				"No FRET events detected. "
				"Inspect picks with Show FRET Traces "
				"and make sure to have FRET events."
			)

		fig1 = plt.figure()
		plt.hist(np.hstack(fret_events), bins=np.arange(0, 1, 0.02))
		plt.title(r"Distribution of $\frac{I_A}{I_D+I_A}$")
		plt.xlabel("Ratio")
		plt.ylabel("Counts")
		fig1.show()

		base, ext = os.path.splitext(
			self.window.view.locs_paths[channel_acceptor]
		)
		out_path = base + ".fret.txt"

		path, ext = QtWidgets.QFileDialog.getSaveFileName(
			self.window.view,
			"Save FRET values as txt and picked locs",
			out_path,
			filter="*.fret.txt"
		)

		if path:
			np.savetxt(
				path,
				np.hstack(fret_events),
				fmt="%1.5f",
				newline="\r\n",
				delimiter="   ",
			)

			locs = stack_arrays(all_fret_locs, asrecarray=True, usemask=False)
			if locs is not None:
				base, ext = os.path.splitext(path)
				out_path = base + ".hdf5"
				pick_info = {"Generated by:": "Picasso Render FRET"}
				io.save_locs(
					out_path,
					locs,
					self.window.view.infos[channel_acceptor] + [pick_info]
				)