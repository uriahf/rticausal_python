# rticausal_python

Python implementation of rticausal for external validation and evaluation of predictions under interventions.

Import package:

```python
import rticausal
```

Initial public API:

```python
rticausal.create_calibration_curve()
rticausal.create_summary_report()
```

`rticausal_python` reuses the existing `rtichoke_python` calibration preparation and rendering machinery. For intervention-specific calibration, the caller supplies predicted risks under each intervention, observed treatment assignments, observed outcomes, and optional identification/design weights. The package does not fit propensity models.

The initial scope is static binary discrete calibration. `create_summary_report()` contains calibration only.
