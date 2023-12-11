# badcad

a jupyter based cad workflow

most of the hard work is done by
- [manifold](https://github.com/elalish/manifold) - constructive solid geometry 
- [pythreejs](https://github.com/jupyter-widgets/pythreejs) - jupyter 3d previews

this project aims to focus on usability in a jupyter environment

first step is finish wrapping / exposing the manifold3d APIs

next step work out a nice pattern for procedural mesh creation

# setup

```bash
pip install git+https://github.com/wrongbad/badcad.git
```

# example

```py
import badcad as bad

bad.sphere(1,64) - bad.sphere(1,64).translate(1,0,0)
```

![spheres](spheres.png)

# vscode

to remove ugly white padding in vs-code jupyter, add a cell like this 
```html
%%html
<style> .cell-output-ipywidget-background { background-color: transparent !important; } </style>
```