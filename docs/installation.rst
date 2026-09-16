Installation
=============

For users
---------

Install the latest release from PyPI:

.. code-block:: bash

   pip install polkupy

polkupy requires Python 3.11 or newer.


For developers
--------------

polkupy uses `pixi <https://pixi.sh>`__. To clone the repository and install
all pixi environments:

.. code-block:: bash

   git clone https://github.com/liammegill/polkupy.git
   cd polkupy
   pixi install --all

This installs polkupy itself in editable mode, so changes to
``src/polkupy`` take effect immediately without reinstalling.

The pixi configuration in ``pyproject.toml`` defines a set of useful
development tasks. Some examples:

.. code-block:: bash

   pixi run -e dev test         # run the pytest suite
   pixi run -e dev typecheck    # mypy
   pixi run -e dev lint         # ruff check
   pixi run -e dev full-check   # typecheck + lint-fix + format
   pixi run -e dev docs-build   # build this site -> docs/_build/html
