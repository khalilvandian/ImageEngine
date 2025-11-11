import marimo

__generated_with = "0.17.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    input_type = mo.ui.dropdown(
        ["Folder", "File", "Multiple Files"],
        value="Folder",
        label="Select Input Type"
    )
    return


if __name__ == "__main__":
    app.run()
