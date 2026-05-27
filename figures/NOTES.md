# Figures — Notes

## draw.io: white vs. blue hyperlink text

**Symptom:** Some italic hyperlink labels render white in draw.io while others render blue.

**Root cause:** When a cell is edited while the browser or OS is in dark mode, draw.io (via the browser clipboard) may bake an inline `light-dark()` CSS color onto the `<i>` tag:

```xml
<i style="background-color: transparent; color: light-dark(rgb(0, 0, 0), rgb(255, 255, 255));">
```

The `light-dark()` CSS function is color-scheme-aware: it resolves to black in light mode and **white in dark mode**. Cells without this inline style inherit draw.io's default hyperlink color (blue), which is why the two groups look different.

**Fix:** Remove the `style="..."` attribute from the `<i>` tags so all cells fall back to the default link color. In the raw XML, replace:

```
&lt;i style=&quot;background-color: transparent; color: light-dark(rgb(0, 0, 0), rgb(255, 255, 255));&quot;&gt;
```

with:

```
&lt;i&gt;
```

This was applied to `Fig1_dataSchema_withLinks_20260527.drawio.xml` (cells `id=44`, `id=65`, `id=78`, `id=108`) on 2026-05-27.

**Prevention:** Edit draw.io diagrams with the browser/OS in light mode, or check the raw XML for `light-dark(` after editing to catch accidental inline color injection.
