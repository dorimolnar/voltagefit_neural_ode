from invoke import task
from pathlib import Path

basepath = r"/Users/moldor/Documents/Thesis/voltagefit_neural_ode/voltage_fitting"

open_cmd = "open"

fig_names = {
    "1": "paper/data",
    "2": "paper/grid_search_fit",
    "3": "paper/no_diff",
    "4": "paper/u_diff",
    "5": "paper/fit_to_experiment",
    "6": "paper/multi_fitting",
    "7": "paper/pospischil",
}

@task
def convertpngpdf(c, fig):
    _convertsvg2pdf(c, fig)
    _convertpdf2png(c, fig)


########################################################################################
# Helpers
########################################################################################
@task
def _convertsvg2pdf(c, fig):
    if fig is None:
        for f in range(len(fig_names)):
            _convert_svg2pdf(c, str(f + 1))
        return
    pathlist = Path(f"{basepath}/{fig_names[fig]}/fig/").glob("*.svg")
    for path in pathlist:
        output_file = str(path).replace(".svg", ".pdf")
        #c.run(f"inkscape {str(path)} --export-pdf={str(path)[:-4]}.pdf")
        c.run(f'inkscape "{str(path)}" --export-type=pdf --export-filename="{output_file}"')


@task
def _convertpdf2png(c, fig):
    if fig is None:
        for f in range(len(fig_names)):
            _convert_pdf2png(c, str(f + 1))
        return
    pathlist = Path(f"{basepath}/{fig_names[fig]}/fig/").glob("*.pdf")
    for path in pathlist:
        output_file = str(path).replace(".pdf", ".png")
        c.run(
            #f'inkscape {str(path)} --export-png={str(path)[:-4]}.png -b "white" --export-dpi=250'
            f'inkscape "{str(path)}" --export-type=png --export-filename="{output_file}" --export-dpi=250'# --background="white"'
        )