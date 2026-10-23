# SWE-Atlas Refactoring task-696719205599a51110d4b458

The `PluginMeta` kind in `apps/plugins/` has a redundant naming problem. Since it already lives inside the plugins package, the "Plugin" prefix creates stuttering like `plugins.PluginMeta`. The plural form `pluginsmeta` is also awkward and has been causing issues with the app SDK codegen.

Can you rename `PluginMeta` to just `Meta` throughout the codebase? This includes the CUE schema definition in `apps/plugins/kinds/` (the hardcoded `machineName` override should be removed so the plural derives naturally from the kind name, the same way the SDK already derives `plugins` from `Plugin`), all the generated Go types and source filenames in the v0alpha1 package, and every consumer that references the old names including the app manifest, the custom storage layer, the metadata providers that convert internal plugin data to the API types, and the RBAC configuration (action string constants whose resource segment currently uses the old plural `pluginsmeta` and must reflect the new naturally derived plural, scope providers whose prefix must use the new plural, role definitions whose fixed role names embed the resource plural, the permission registry kind scope prefix mappings, and the auth mapper resource translation which maps API resource names to internal RBAC prefixes).

For instance, `PluginMetaKind()` becomes `MetaKind()`, `NewPluginMetaClient()` becomes `NewMetaClient()`, `PluginMetaJSONData` becomes `MetaJSONData`, `jsonDataToPluginMetaJSONData` becomes `jsonDataToMetaJSONData`, `PluginMetaStorage` becomes `MetaStorage`, and on the RBAC side the resource segment changes too, so `plugins.pluginsmeta:create` becomes `plugins.metas:create`. Search the full codebase for all remaining `PluginMeta` references and rename them following the same pattern.

While you're at it, both `Meta` and `Plugin` are missing proper constructors that initialize default Spec and Status values. Add those constructors so that each nested subtype is recursively initialized with its own schema defaults (for example, empty slices as `[]` rather than `nil`, nested structs via their own constructors). Use the Plugin constructor in its schema registration instead of a bare struct literal.

I've already taken care of all changes to the test files. Do NOT modify any test files or testing logic in any way. Your task is to make the minimal changes to source files only.

Use the below Interface:

- Path: `apps/plugins/pkg/apis/plugins/v0alpha1/meta_object_gen.go`
- Name: `NewMeta`
- Type: function
- Input: `NA`
- Output: `*Meta`
- Description: Creates a new Meta resource object with properly initialized Spec and Status values, where each nested subtype is recursively initialized with its own schema defaults.

- Path: `apps/plugins/pkg/apis/plugins/v0alpha1/plugin_object_gen.go`
- Name: `NewPlugin`
- Type: function
- Input: `NA`
- Output: `*Plugin`
- Description: Creates a new Plugin resource object with properly initialized Spec and Status values, where each nested subtype is recursively initialized with its own schema defaults.


Official task tag: `Go`.

This task card does not contain the project checkout. Use the official task container/image for the exact base workspace and validation. Do not search online for a reference solution.
