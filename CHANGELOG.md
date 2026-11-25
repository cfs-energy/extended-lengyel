# Changelog

## Unreleased

### cfspopcon plugin
- `extended_lengyel` is a cfspopcon 9 plugin. Register it with `cfspopcon.register_plugin("extended_lengyel")`; see cfspopcon's [Authoring a Plugin](https://cfspopcon.readthedocs.io/en/latest/doc_sources/authoring_a_plugin.html) page. The `xlengyel` command line interface always registers it, in addition to any other plugins listed under `plugins` in the input file.
- The convective-layer fit in `curve_fit.yml` is regenerated with the new `xlengyel-fit-convective-layer` command (`extended_lengyel/fit_convective_layer.py`) instead of by running `notebooks/4.Studying_the_convective_layer.ipynb`, which no longer writes the file.
- Default units are declared in `extended_lengyel/variables.yaml`, which replaces `extended_units.yaml`. Algorithms use `declare_algorithm` and `CompositeAlgorithm.declare`; look them up with `cfspopcon.registry["name"]` instead of `Algorithm.get_algorithm`.
- Requires cfspopcon `^9.0.2` and radas `^1!2.0.1`.
