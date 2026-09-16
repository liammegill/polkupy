polkupy
=======

The polkupy library is designed to help you work with, analyse and visualise
cycling data. It can directly parse common data sources, including GPX and FIT.

polkupy provides the the :class:`~polkupy.Ride` class, which is a chainable
`pandas <https://pandas.pydata.org>`__-backed object with a rich Jupyter
display. Multiple Rides together constitute a :class:`~polkupy.Trips` class.
Both can be filtered and mutated, allowing for comparisons between bikes,
riders and routes.

View the source code on `GitHub <https://github.com/liammegill/polkupy>`__.


.. toctree::
   :maxdepth: 1

   installation
   quickstart
   api
