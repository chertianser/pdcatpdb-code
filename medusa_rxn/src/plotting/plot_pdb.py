from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib import colormaps
from sklearn.linear_model import LinearRegression
from typing import List
from math import ceil


class PlotPDB():
    def __init__(self, dropna=False):
        self.dropna = dropna
        self.load_data(dropna=dropna)


    def load_data(self, dropna=None):
        self.base_dir = Path('.')
        self.data_source_file = self.base_dir/"supplementary_data_s1.xlsx"
        self.table_titles = ["calibration curv", "PDB vs CPL", "PDB vs Phosphine",
                             "PDB_JP vs Base", "PDB_CyJP vs Base", "PDB_noP vs Base", "PDB vs Base",
                             "kinetics Bpin raw", "kinetics BOH2 raw", "kinetics summary",
                             "kinetics Bpin K3PO4 raw", "kinetics summary K3PO4",
                             "4 L 0 H2O", "4 L 0 H2O NaphBeg", "4 L 20 H2O", "4 L 100 H2O",
                             "PCy3 0 20 100 H2O", "PtBu3 0 20 100 H2O", "CyJP 0 20 100 H2O", "JP 0 20 100 H2O",
                             "vary Pd load", "JP eq", "PdOAc2 Pd2dba3 Buchwald"
                             ]
        self.data_all = {}
        for table_title in self.table_titles:
            data = pd.read_excel(self.data_source_file, sheet_name=table_title)
            if dropna == True:
                self.data_all[table_title] = data.dropna()
            else:
                self.data_all[table_title] = data

    @staticmethod
    def format_coefficient(value):
        exp = int(np.floor(np.log10(abs(value))))
        coeff = value / 10**exp
        return f"{coeff:.2f} \\times 10^{{{exp}}}"

    def plot_scatter_err_fit(self, table_title,
                             x_name, y_name, x_error=None, y_error=None,
                             linear_fitting = [],
                             # here it contains the m-th and n-th elements to be used for linear fitting
                             saveas = None):

        params = {
            "font.family": "Arial",
            "mathtext.default": "regular"
        }
        plt.rc("font", size=12)
        plt.rcParams.update(params)
        plt.rcParams["axes.facecolor"] = "white"

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)

        x = self.data_all[table_title][x_name].values.reshape(-1, 1)
        y = self.data_all[table_title][y_name].values

        xerr = None
        yerr = None
        if x_error:
            xerr = self.data_all[table_title][x_error]
        if y_error:
            yerr = self.data_all[table_title][y_error]
        ax.errorbar(x, y, xerr = xerr, yerr = yerr, fmt='o')
        if linear_fitting:
            model = LinearRegression()
            model.fit(x[linear_fitting[0]:linear_fitting[1]], y[linear_fitting[0]:linear_fitting[1]])
            y_fit = model.predict(x)
            ax.plot(x, y_fit, linestyle = '--')

            # eq = ""
            # eq for calibration curve
            # eq = ("$\\frac{%s_{%s}}{%s_{%s}} = %s \\ \\frac{%s_{%s}}{%s_{%s}}"
            #       % (y_name.split('/')[0].split('_')[0], y_name.split('/')[0].split('_')[1],
            #          y_name.split('/')[1].split('_')[0], y_name.split('/')[1].split('_')[1],
            #          self.format_coefficient(model.coef_[0]),
            #          x_name.split('/')[0].split('_')[0], x_name.split('/')[0].split('_')[1],
            #          x_name.split('/')[1].split('_')[0], x_name.split('/')[1].split('_')[1],
            #          ))

            # eq for kinetic moniter curve
            # eq = ("$%s_{%s} = %s \\ %s"
            #       % (y_name.split('_')[0], y_name.split('_')[1][:-2],
            #          self.format_coefficient(model.coef_[0]),
            #          x_name.split('/')[0]
            #          )
            #       )

            eq = "$\\it{ln(k)} = \\frac{%s}{T} " % self.format_coefficient(model.coef_[0])
            if model.intercept_ >= 0:
                eq += "+%s$" % self.format_coefficient(model.intercept_)
            else:
                eq += "%s$" % self.format_coefficient(model.intercept_)
            eq += "\t$R^2 = %.4f$" % model.score(x[linear_fitting[0]:linear_fitting[1]],
                                                 y[linear_fitting[0]:linear_fitting[1]])

            ax.legend([eq])
        # axes label for calibration curve
        # ax.set_xlabel("$(%s_{%s}\\ /\\ %s_{%s})$" % (x_name.split('/')[0].split('_')[0], x_name.split('/')[0].split('_')[1],
        #                                        x_name.split('/')[1].split('_')[0], x_name.split('/')[1].split('_')[1]))
        # ax.set_ylabel("$(%s_{%s}\\ /\\ %s_{%s})$" % (y_name.split('/')[0].split('_')[0], y_name.split('/')[0].split('_')[1],
        #                                        y_name.split('/')[1].split('_')[0], y_name.split('/')[1].split('_')[1]))

        # ax.set_xlabel("time / min")
        # ax.set_ylabel("Naph / %")
        # ax.set_ylim(-0.5, ceil(max(y)))

        ax.set_xlabel("$\\frac{1}{T} \\ /\\ K^{-1}$")
        ax.set_ylabel("$\\it{ln(k_{init})}$")


        if saveas:
            plt.savefig(self.base_dir/'figs'/f'{saveas}.svg', format="svg")
            plt.savefig(self.base_dir / 'figs' / f'{saveas}.png', format="png")
        else:
            plt.show()
        plt.clf()

    def plot_scatter_err_multi(self, table_title, x_name, y_names, y_errors=None, y_axis_name = "", saveas = None):
        colors = plt.get_cmap("tab10")
        for i in range(0, len(y_names)):
            plt.errorbar(x=self.data_all[table_title][x_name],
                         y=self.data_all[table_title][y_names[i]],
                         yerr=self.data_all[table_title][y_errors[i]],
                         fmt='o',
                         markersize=5,
                         color=colors(i),
                         label=y_names[i]
                         )
        plt.xlabel(x_name)
        plt.ylabel(y_axis_name)
        plt.legend()
        plt.ylim((0,100))
        # ax.set_title(f"Calibration curve for {y_name} vs {x_name}")
        if saveas:
            plt.savefig(self.base_dir/'figs'/f'{saveas}.svg', format="svg")
        else:
            plt.show()
        plt.clf()

    def plot_bar_err_simple(self, table_title, x_name, y_name, y_error=None, x_tick_rot = None, saveas=None):
        fig, ax = plt.subplots() #figsize=(8,6))
        colors = plt.get_cmap("tab10")
        x_name_str = [str(x) for x in self.data_all[table_title][x_name].values]
        ax.bar(x = x_name_str,
               height = self.data_all[table_title][y_name].values,
               yerr = self.data_all[table_title][y_error].values,
               color = colors(0),
               label = x_name_str
               )
        # ax.set_xticks(x_name_str)
        ax.set_xticklabels(ax.get_xticklabels(), rotation = x_tick_rot)
        ax.set_ylim((0,100))
        ax.set_ylabel(y_name)
        ax.set_xlabel(x_name)

        # plt.subplots_adjust(bottom = 0.5, top=0.95)
        if saveas:
            plt.savefig(self.base_dir/'figs'/f'{saveas}.svg', format="svg")
        else:
            plt.show()
        plt.clf()

    def plot_bar_err_multi(self, table_title, x_name, y_names,
                           y_errors=None, x_tick_rot=None, saveas=None,
                           fig_width = 6, fig_height = 6, from_bottom=0
                           ):
        params = {
            "font.family": "Arial",
            "mathtext.default": "regular"
        }
        plt.rc("font", size=16)
        plt.rcParams.update(params)
        plt.rcParams["axes.facecolor"] = "white"
        fix, ax = plt.subplots(figsize=(fig_width,fig_height))
        colors = plt.get_cmap("tab10")
        x_name_str = [str(x) for x in self.data_all[table_title][x_name].values]
        x_num = len(x_name_str)
        x_range = np.arange(x_num)
        y_num = len(y_names)
        width = 0.6 * fig_width/ (x_num * y_num)
        for i in range (0, y_num):
            ax.bar(x = (x_range*fig_width/x_num + width * i),
                   height = self.data_all[table_title][y_names[i]].values,
                   width = width,
                   yerr = self.data_all[table_title][y_errors[i]].values,
                   color = colors(i),
                   label = y_names[i]
                   )
        ax.set_xticks((x_range + width/2)*fig_width/x_num, labels = x_name_str, rotation = x_tick_rot)
        ax.set_xlabel(x_name)
        ax.legend()
        ax.set_ylim((0, 100))
        plt.subplots_adjust(bottom=from_bottom, top=0.95, left = 0.05, right = 0.98)
        # plt.figure(figsize=(10, 6))
        if saveas:
            plt.savefig(self.base_dir/'figs'/f'{saveas}.svg', format="svg")
            plt.savefig(self.base_dir / 'figs' / f'{saveas}.png')
        else:
            plt.show()
        plt.clf()

    def plot_bar_err_multi_grouping(self, table_title, x_name, y_name, x_group_nums,
                                    y_error=None, x_tick_rot=None, saveas=None,
                                    fig_width = 6, fig_height = 6, from_bottom=0
                                    ):
        params = {
            "font.family": "Arial",
            "mathtext.default": "regular"
        }
        plt.rc("font", size=16)
        plt.rcParams.update(params)
        plt.rcParams["axes.facecolor"] = "white"
        fix, ax = plt.subplots(figsize=(fig_width, fig_height))
        colors = plt.get_cmap("tab10")
        color_map = []
        index = 0
        for group_size in x_group_nums:
            color_map.extend([colors(index)]*group_size)
            index += 1

        x_name_str = [str(x) for x in self.data_all[table_title][x_name].values]
        x_num = len(x_name_str)
        x_range = np.arange(x_num)
        width = 0.6 * fig_width / x_num

        ax.bar(x = x_range*fig_width/x_num,
               height = self.data_all[table_title][y_name].values,
               width = width,
               yerr = self.data_all[table_title][y_error].values,
               color = color_map,
               label = y_name
               )
        ax.set_xticks((x_range + width/2)*fig_width/x_num, labels = x_name_str, rotation = x_tick_rot)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        # ax.legend()
        ax.set_ylim((0, 20))
        plt.subplots_adjust(bottom=from_bottom, top=0.95, left = 0.1, right = 0.98)
        # plt.figure(figsize=(10, 6))
        if saveas:
            plt.savefig(self.base_dir/'figs'/f'{saveas}.svg', format="svg")
            plt.savefig(self.base_dir / 'figs' / f'{saveas}.png')
        else:
            plt.show()
        plt.clf()

    def plot_CPL_PDB(self, saveas=None):
        plt.rcParams.update({'font.size': 14})
        data = self.data_all["PDB vs CPL"]
        data_60 = data[data["temperature"]==60]
        data_100 = data[data["temperature"]==100]
        fig, axs = plt.subplots(2, 2, figsize=(16, 15))
        pd=["Pd(OAc)2", "Pd2dba3"]
        phos=["PPh3","PCy3","PtBu3","CyJohnPhos","JohnPhos"]
        h2o=[0,20]
        l_eq=[1,2]
        x_pos = np.array([0, 4, 8, 12, 16])
        colours = plt.get_cmap("tab10")
        for i in range(0, 2):
            for j in range(0, 2):
                data_mini = data_60[(data_60["Pd source"] == pd[i]) & (data_60["H2O equiv"] == h2o[j])]
                data_1_phos = data_mini[data_mini["phosphine equiv"] == 1]
                data_2_phos = data_mini[data_mini["phosphine equiv"] == 2]
                axs[i, j].bar(x_pos, width=0.3, height=0, yerr=0, color=colours(9), label="CPL, no Pd")
                axs[i, j].bar(x_pos+0.4, width=0.3, height=0, yerr=0, color=colours(10), label="PDB, no Pd")

                axs[i, j].bar(x_pos+0.9, width=0.3, height=data_1_phos["CPL%"], yerr=data_1_phos["err(CPL)%"],
                              color=colours(0), label="CPL, 1 eq L")
                axs[i, j].bar(x_pos+1.3, width=0.3, height=data_1_phos["PDB%"], yerr=data_1_phos["err(PDB)%"],
                              color=colours(1), label="PDB, 1 eq L")

                axs[i, j].bar(x_pos+1.8, width=0.3, height=data_2_phos["CPL%"], yerr=data_2_phos["err(CPL)%"],
                              color=colours(2), label="CPL, 2 eq L")
                axs[i, j].bar(x_pos+2.2, width=0.3, height=data_2_phos["PDB%"], yerr=data_2_phos["err(PDB)%"],
                              color=colours(3), label="PDB, 2 eq L")

                axs[i, j].bar(x_pos + 2.7, width=0.3, height=data_100["CPL%"], yerr=data_100["err(CPL)%"],
                              color=colours(4), label="CPL, 1 eq L, 100 C")
                axs[i, j].bar(x_pos + 3.1, width=0.3, height=data_100["PDB%"], yerr=data_100["err(PDB)%"],
                              color=colours(5), label="PDB, 1 eq L, 100 C")


                axs[i, j].set_ylim(0,100)
                # axs[i, j].legend()
                axs[i, j].set_xticks(x_pos+1, phos)
                axs[i, j].set_title(f"{pd[i]}, {h2o[j]} eq added H2O")
        handles, labels = axs[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4)
        fig.tight_layout(rect=[0.02, 0.05, 0.98,0.98])
        if saveas:
            plt.savefig(self.base_dir / 'figs' / f'{saveas}.png')
            plt.savefig(self.base_dir / 'figs' / f'{saveas}.svg', format="svg")
        else:
            plt.show()
        plt.clf()

    def plot_vary_Pd_load(self, saveas=None):
        colors = plt.get_cmap("tab10")
        # color_orange = plt.get_cmap("tab20c")
        # colours=plt.get_cmap("tab20")
        data = self.data_all["vary Pd load"]
        fig, ax = plt.subplots(figsize=(18, 12))
        x_pos = np.array([0, 6, 12])
        pd_load = data["Pd/P mol%"].unique()
        phosphines = data["phosphine"].unique()
        multiplier = 0
        grid=0.9
        for phosphine in phosphines:
            offset = grid*multiplier
            ax.bar(x_pos+offset, width=0.3, height=data[data["phosphine"] == phosphine]["Naph_yield"],
                   yerr=data[data["phosphine"] == phosphine]["Naph_err"], label=phosphine + ", Naph", color=colors(multiplier))
            # ax.bar(x_pos + offset +0.35, width=0.3, height=data[data["phosphine"] == phosphine]["biNaph_yield"],
            #        yerr=data[data["phosphine"] == phosphine]["biNaph_err"], label=phosphine + ", biNaph", color=color_orange(multiplier))

            # ax.bar(x_pos + offset, width=0.6, height=data[data["phosphine"] == phosphine]["Naph_yield"],
            #        yerr=data[data["phosphine"] == phosphine]["Naph_err"], label=phosphine + ", Naph",
            #        color=colors(multiplier))
            # ax.bar(x_pos + offset, width=0.6, height=data[data["phosphine"] == phosphine]["biNaph_yield"],
            #        yerr=data[data["phosphine"] == phosphine]["biNaph_err"], label=phosphine + ", biNaph",
            #        color=color_blue(multiplier))

            multiplier += 1
        ax.set_xticks(x_pos + 2, [f"{x} mol%" for x in pd_load])
        # ax.set_xlabel([f"{x} mol%" for x in pd_load])
        ax.set_ylim(0, 100)
        ax.legend()
        if saveas:
            plt.savefig(self.base_dir / 'figs' / f'{saveas}.png')
            plt.savefig(self.base_dir / 'figs' / f'{saveas}.svg', format="svg")
        else:
            plt.show()
        plt.clf()

    def plot_base_comb(self, saveas = None):
        colors = plt.get_cmap("tab10")
        params = {
            "font.family": "Arial",
            "mathtext.default": "regular"
        }
        plt.rc("font", size=18)
        plt.rcParams.update(params)
        plt.rcParams["axes.facecolor"] = "white"
        fig, ax = plt.subplots(figsize=(15, 10))

        data = self.data_all["PDB vs Base"]
        bases = data["base"].unique()
        phosphines = data["phosphine"].unique()

        multiplier = 0
        width = 0.25

        for phosphine in phosphines:
            ax.bar(np.arange(len(bases)) + multiplier*width, width = width,
                   height = data[data["phosphine"]==phosphine]["BiNaph / %"], yerr = data[data["phosphine"]==phosphine]["err(BiNaph) / %"],
                   label = phosphine, color = colors(multiplier))
            multiplier += 1
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
        ax.set_xticks(np.arange(len(bases))+0.3, bases)
        ax.set_xlabel("Bases")
        ax.set_ylabel("BiNaph / %")
        ax.set_ylim(0, 10)
        plt.legend()
        fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.98])
        if saveas:
            plt.savefig(self.base_dir / 'figs' / f'{saveas}.png')
            plt.savefig(self.base_dir / 'figs' / f'{saveas}.svg', format="svg")
        else:
            plt.show()
        plt.clf()

    def plot_loading(self, saveas = None):
        colors = plt.get_cmap("tab10")
        params = {
            "font.family": "Arial",
            "mathtext.default": "regular"
        }
        plt.rc("font", size=16)
        plt.rcParams.update(params)
        plt.rcParams["axes.facecolor"] = "white"
        fig, ax = plt.subplots(figsize=(12, 10))
        data = self.data_all["vary Pd load"]
        loadings = data["Pd/P mol%"].unique()
        phosphines = data["phosphine"].unique()
        multiplier = 0
        width = 0.25
        for loading in loadings:
            ax.bar(np.arange(len(phosphines)) + multiplier * width, width=width,
                   height=data[data["Pd/P mol%"] == loading]["Naph / %"],
                   yerr=data[data["Pd/P mol%"] == loading]["err(Naph) / %"],
                   label=loading, color=colors(multiplier))
            multiplier += 1
        ax.set_xticklabels(ax.get_xticklabels(), rotation=75)
        ax.set_xticks(np.arange(len(loadings)) + 0.3, loadings)
        ax.set_xlabel("Bases")
        ax.set_ylabel("Naph / %")
        ax.set_ylim(0, 100)
        plt.legend()
        fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.98])
        if saveas:
            plt.savefig(self.base_dir / 'figs' / f'{saveas}.png')
            plt.savefig(self.base_dir / 'figs' / f'{saveas}.svg', format="svg")
        else:
            plt.show()
        plt.clf()


pdb = PlotPDB(dropna=False)
# kinetic monitor
pdb.plot_scatter_err_fit(table_title="kinetics summary K3PO4", x_name="1/T (K-1)", y_name="ln(k)_avg", y_error="ln(k)_err",
                         linear_fitting=[0,4],
                         saveas="Kinetic fitting"
                         )

pdb.plot_scatter_err_fit(table_title="kinetics Bpin K3PO4 raw", x_name="time / min", y_name="Naph_328K 1",
                         linear_fitting=[5,14],
                         saveas="Kinetic monitoring 328K 1"
                         )
pdb.plot_scatter_err_fit(table_title="kinetics Bpin K3PO4 raw", x_name="time / min", y_name="Naph_328K 2",
                         linear_fitting=[5,14],
                         saveas="Kinetic monitoring 328K 2"
                         )
pdb.plot_scatter_err_fit(table_title="kinetics Bpin K3PO4 raw", x_name="time / min", y_name="Naph_333K 1",
                         linear_fitting=[4, 9],
                         saveas="Kinetic monitoring 333K 1"
                         )
pdb.plot_scatter_err_fit(table_title="kinetics Bpin K3PO4 raw", x_name="time / min", y_name="Naph_333K 2",
                         linear_fitting=[4, 9],
                         saveas="Kinetic monitoring 333K 2"
                         )
pdb.plot_scatter_err_fit(table_title="kinetics Bpin K3PO4 raw", x_name="time / min", y_name="Naph_333K 3",
                         linear_fitting=[4, 9],
                         saveas="Kinetic monitoring 333K 3"
                         )
pdb.plot_scatter_err_fit(table_title="kinetics Bpin K3PO4 raw", x_name="time / min", y_name="Naph_338K 1",
                         linear_fitting=[4, 9],
                         saveas="Kinetic monitoring 338K 1"
                         )
pdb.plot_scatter_err_fit(table_title="kinetics Bpin K3PO4 raw", x_name="time / min", y_name="Naph_338K 2",
                         linear_fitting=[4, 9],
                         saveas="Kinetic monitoring 338K 2"
                         )
pdb.plot_scatter_err_fit(table_title="kinetics Bpin K3PO4 raw", x_name="time / min", y_name="Naph_343K 1",
                         linear_fitting=[5, 9],
                         saveas="Kinetic monitoring 343K 1"
                         )
pdb.plot_scatter_err_fit(table_title="kinetics Bpin K3PO4 raw", x_name="time / min", y_name="Naph_343K 2",
                         linear_fitting=[5, 9],
                         saveas="Kinetic monitoring 343K 2"
                         )

# PDB JohnPhos load

pdb.plot_bar_err_multi(table_title="JP eq",
                       x_name = "JohnPhos eq",
                       y_names = ["Naph / %"],
                       y_errors = ["err(Naph) / %"],
                       x_tick_rot=0,
                       fig_width=10,
                       fig_height=10,
                       from_bottom=0.1,
                       saveas = "PDB with Pd(OAc)2, different JohnPhos eq"
                       )

# PDB vs base

pdb.plot_base_comb("PDB vs base combined BiNaph rerange")
pdb.plot_bar_err_multi_grouping(table_title="PDB vs Base",
                                x_name = "base",
                                y_name = "Naph / %",
                                x_group_nums=[17,11,11],
                                y_error = "err(Naph) / %",
                                x_tick_rot = 90,
                                fig_width = 15,
                                fig_height = 10,
                                from_bottom = 0.25,
                                saveas = "PDB vs Base, Naph only"
                                )
pdb.plot_bar_err_multi_grouping(table_title="PDB vs Base",
                                x_name = "base",
                                y_name = "BiNaph / %",
                                x_group_nums=[17,11,11],
                                y_error = "err(BiNaph) / %",
                                x_tick_rot = 90,
                                fig_width = 15,
                                fig_height = 10,
                                from_bottom = 0.25,
                                saveas = "PDB vs Base, BiNaph only"
                                )
pdb.plot_bar_err_multi(table_title="PDB_JP vs Base",
                       x_name = "base",
                       y_names = ["Naph / %", "BiNaph / %"],
                       y_errors = ["err(Naph) / %", "err(BiNaph) / %"],
                       x_tick_rot=75,
                       fig_width=15,
                       fig_height=10,
                       from_bottom=0.25,
                       saveas = "Protodeboronation with Pd(OAc)2+JohnPhos, vary base"
                       )
pdb.plot_bar_err_multi(table_title="PDB_JP vs Base",
                       x_name = "base",
                       y_names=["Naph / %"],
                       y_errors=["err(Naph) / %"],
                       x_tick_rot=75,
                       fig_width=15,
                       fig_height=10,
                       from_bottom=0.25,
                       saveas = "Protodeboronation with Pd(OAc)2+JohnPhos, vary base, Naph only"
                       )
pdb.plot_bar_err_multi(table_title="PDB_JP vs Base",
                       x_name = "base",
                       y_names=["BiNaph / %"],
                       y_errors=["err(BiNaph) / %"],
                       x_tick_rot=75,
                       fig_width=15,
                       fig_height=10,
                       from_bottom=0.25,
                       saveas = "Protodeboronation with Pd(OAc)2+JohnPhos, vary base, BiNaph only"
                       )

# vary Pd load

pdb.plot_bar_err_multi_grouping(table_title="vary Pd load",
                                x_name = "phosphine",
                                y_name = "Naph / %",
                                x_group_nums=[6,6,6],
                                y_error = "err(Naph) / %",
                                x_tick_rot = 75,
                                fig_width = 15,
                                fig_height = 10,
                                from_bottom = 0.25,
                                saveas = "Vary Pd loading Naph"
                                )
pdb.plot_bar_err_multi_grouping(table_title="vary Pd load",
                                x_name = "phosphine",
                                y_name = "BiNaph / %",
                                x_group_nums=[6,6,6],
                                y_error = "err(BiNaph) / %",
                                x_tick_rot = 75,
                                fig_width = 15,
                                fig_height = 10,
                                from_bottom = 0.25,
                                saveas = "Vary Pd loading BiNaph rerange"
                                )

# PDB vs Phosphine extended

pdb.plot_bar_err_multi(table_title="PDB vs Phosphine",
                       x_name = "phosphine",
                       y_names = ["Naph / %", "BiNaph / %"],
                       y_errors = ["err(Naph) / %", "err(BiNaph) / %"],
                       x_tick_rot = 75,
                       fig_width = 15,
                       fig_height = 10,
                       from_bottom = 0.25,
                       saveas = "Protodeboronation with Pd(OAc)2+K3PO4, vary phosphine"
                       )
pdb.plot_bar_err_multi(table_title="PDB vs Phosphine",
                       x_name = "phosphine",
                       y_names=["Naph / %"],
                       y_errors=["err(Naph) / %"],
                       x_tick_rot=75,
                       fig_width=15,
                       fig_height=10,
                       from_bottom=0.25,
                       saveas = "Protodeboronation with Pd(OAc)2+K3PO4, vary phosphine, Naph only"
                       )
pdb.plot_bar_err_multi(table_title="PDB vs Phosphine",
                       x_name = "phosphine",
                       y_names=["BiNaph / %"],
                       y_errors=["err(BiNaph) / %"],
                       x_tick_rot=75,
                       fig_width=15,
                       fig_height=10,
                       from_bottom=0.25,
                       saveas = "Protodeboronation with Pd(OAc)2+K3PO4, vary phosphine, BiNaph only, rerange"
                       )

# pdb [Pd] L Buchwald

pdb.plot_bar_err_multi_grouping(table_title="PdOAc2 Pd2dba3 Buchwald",
                                x_name = "Pd-phosphine complex",
                                y_name = "Naph / %",
                                x_group_nums=[6,6],
                                y_error = "err(Naph) / %",
                                x_tick_rot = 75,
                                fig_width = 12,
                                fig_height = 12,
                                from_bottom = 0.3,
                                saveas = "Protodeboronation of 2-NaphBpin with [Pd]+L and preformed LPd cycle"
                                )



# plot 4 ligands over 3 water loading

pdb.plot_scatter_err_multi(table_title="4 L 0 H2O",
                           x_name="time (h)",
                           y_names=["PCy3_yield", "PtBu3_yield", "CyJohnPhos_yield", "JohnPhos_yield"],
                           y_errors=["PCy3_error", "PtBu3_error", "CyJohnPhos_error", "JohnPhos_error"],
                           y_axis_name="Yield %",
                           saveas="Ligand dependent protodeboronation over time, 0 H2O loading, full scale"
                           )
pdb.plot_scatter_err_multi(table_title="4 L 20 H2O",
                           x_name="time (h)",
                           y_names=["PCy3_yield", "PtBu3_yield", "CyJohnPhos_yield", "JohnPhos_yield"],
                           y_errors=["PCy3_error", "PtBu3_error", "CyJohnPhos_error", "JohnPhos_error"],
                           y_axis_name="Yield %",
                           saveas="Ligand dependent protodeboronation over time, 20 H2O loading, full scale"
                           )
pdb.plot_scatter_err_multi(table_title="4 L 100 H2O",
                           x_name="time (h)",
                           y_names=["PCy3_yield", "PtBu3_yield", "CyJohnPhos_yield", "JohnPhos_yield"],
                           y_errors=["PCy3_error", "PtBu3_error", "CyJohnPhos_error", "JohnPhos_error"],
                           y_axis_name="Yield %",
                           saveas="Ligand dependent protodeboronation over time, 100 H2O loading, full scale"
                           )

pdb.plot_scatter_err_multi(table_title="4 L 0 H2O NaphBeg",
                           x_name="time (h)",
                           y_names=["PCy3_yield", "PtBu3_yield", "CyJohnPhos_yield", "JohnPhos_yield"],
                           y_errors=["PCy3_error", "PtBu3_error", "CyJohnPhos_error", "JohnPhos_error"],
                           y_axis_name="Yield %",
                           saveas="Ligand dependent protodeboronation over time, 0 H2O loading, NaphBeg, full scale"
                           )

pdb.plot_scatter_err_multi(table_title="JP 0 20 100 H2O",
                           x_name="time (h)",
                           y_names=["0eq_H2O_yield", "20eq_H2O_yield", "100eq_H2O_yield"],
                           y_errors=["0eq_H2O_error", "20eq_H2O_error", "100eq_H2O_error"],
                           y_axis_name="Yield %",
                           saveas="Water dependent protodeboronation over time, JohnPhos, full scale"
                           )
pdb.plot_scatter_err_multi(table_title="PCy3 0 20 100 H2O",
                           x_name="time (h)",
                           y_names=["0eq_H2O_yield", "20eq_H2O_yield", "100eq_H2O_yield"],
                           y_errors=["0eq_H2O_error", "20eq_H2O_error", "100eq_H2O_error"],
                           y_axis_name="Yield %",
                           saveas="Water dependent protodeboronation over time, PCy3, full scale"
                           )
pdb.plot_scatter_err_multi(table_title="PtBu3 0 20 100 H2O",
                           x_name="time (h)",
                           y_names=["0eq_H2O_yield", "20eq_H2O_yield", "100eq_H2O_yield"],
                           y_errors=["0eq_H2O_error", "20eq_H2O_error", "100eq_H2O_error"],
                           y_axis_name="Yield %",
                           saveas="Water dependent protodeboronation over time, PtBu3, full scale"
                           )
pdb.plot_scatter_err_multi(table_title="CyJP 0 20 100 H2O",
                           x_name="time (h)",
                           y_names=["0eq_H2O_yield", "20eq_H2O_yield", "100eq_H2O_yield"],
                           y_errors=["0eq_H2O_error", "20eq_H2O_error", "100eq_H2O_error"],
                           y_axis_name="Yield %",
                           saveas="Water dependent protodeboronation over time, CyJohnPhos, full scale"
                           )

calib_pdb = PlotPDB(dropna=True)

# plot calibration curves

calib_pdb.plot_scatter_err_fit(table_title="calibration curv", x_name="m_Naph / m_oTP", y_name="Int_Naph / Int_oTP",
                         linear_fitting=[0,6],
                         saveas="Calibration curve of Naph"
                         )
calib_pdb.plot_scatter_err_fit(table_title="calibration curv", x_name="m_BiNaph / m_oTP", y_name="Int_BiNaph / Int_oTP",
                         linear_fitting=[0,7],
                         saveas="Calibration curve of BiNaph"
                         )
calib_pdb.plot_scatter_err_fit(table_title="calibration curv", x_name="m_CPL / m_oTP", y_name="Int_CPL / Int_oTP",
                         linear_fitting=[0,7],
                         saveas="Calibration curve of CPL"
                         )
calib_pdb.plot_scatter_err_fit(table_title="calibration curv", x_name="m_2-NaphBpin / m_oTP", y_name="Int_2-NaphBpin / Int_oTP",
                         linear_fitting=[0,7],
                         saveas="Calibration curve of 2-NaphBpin"
                         )