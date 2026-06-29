"""Presentation layer: the interactive terminal UI (an outer adapter).

This package is the outermost ring of the Clean Architecture. It depends on the
`application` services (`Trainer`, `PredictionService`) and on the Python standard
library only; nothing in the domain or the inner layers depends on it.

UI conventions (enforced across every module here):
  * The interface is a continuous stream of output: views are printed one after the
    other and the screen is never cleared between them.
  * Input is line-based and always confirmed with Enter; a fixed `> ` prompt is shown
    before the cursor whenever the program waits (see `prompt.py`).
  * No emojis and no Unicode arrow glyphs (`->`/`<-`/`up`/`down` style characters).
  * State is shown with plain ASCII markers: `[ ]`, `[x]`, `[*]`, `[done]`, `-`, `*`,
    moving dots `...`.
  * Colour and emphasis are carried by ANSI styles defined in `ansi.py`, which
    degrade gracefully to plain text when the output is not a colour-capable TTY.
"""
