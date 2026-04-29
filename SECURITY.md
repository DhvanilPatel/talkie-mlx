# Security

Report security issues through GitHub private security advisories when possible.

The converter reads PyTorch `.pt` and `.ckpt` files. Treat model checkpoints as
trusted inputs only, and prefer the official Talkie repositories or local files
you created yourself.

This repository does not include model weights. Converted weights should stay in
local ignored directories unless you intentionally publish them somewhere else.
