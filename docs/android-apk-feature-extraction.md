# Android APK feature extraction

`FileAnalyzer` invokes `AndroidAPKFeatureExtractor` for `.apk` files and ZIP files
containing `AndroidManifest.xml`. The extractor uses Androguard to read declared
manifest permissions and external method references from all parsed DEX files.
It does not execute the APK.

The trained feature schema has 470 names. The first 35 are behavior categories
(for example, `NETWORK_ACCESS____` and `SMS_SEND____`) with no documented static
derivation in this project. The other 435 names are method-like identifiers. The
extractor compares those names with external DEX method references and records
`present` or `absent` when DEX parsing succeeds. The schema has no declaring class
or method signature, so a same-name method in another class can match. The
training dataset's original extraction semantics are not documented here; this
static result must not be represented as an exact reconstruction of those
semantics.

Declared manifest permissions are retained as separate evidence. None of the
470 schema names is an Android permission name, and this project has no supported
permission-to-category mapping, so permissions do not populate the autoencoder
vector. The 35 behavior categories remain `unavailable`; they are not set to 0.
The vector is therefore incomplete and the strict autoencoder layer remains
unavailable. The other five novelty layers continue to consume their existing
features.

For malformed APKs or a missing parser dependency, extraction details use the
`unavailable` state and include the failure reason. For a valid APK without DEX,
method feature states are `unavailable`, not `absent`.

Install the added runtime dependency from the project metadata with:

```powershell
python -m pip install -e .
```

The extractor tests use deterministic parser fixtures because this repository
contains no real APK sample. They validate the adapter and flow, not end-to-end
APK parsing against a real APK.
