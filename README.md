# badcad

A jupyter based CAD workflow

Most of the hard work is done by:
- [manifold](https://github.com/elalish/manifold) - Constructive Solid Geometry 
- [pythreejs](https://github.com/jupyter-widgets/pythreejs) - Jupyter 3D Previews

This project aims to focus on usability in a jupyter environment.

First step is finish exposing the manifold3d.Manifold APIs

Next step work out a nice pattern for procedural mesh creation

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
