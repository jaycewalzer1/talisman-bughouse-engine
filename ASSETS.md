# Third-party assets

Everything else in the 3D view (textures, graffiti, prints, shoes, hair cards, hats, props) is generated in code and is
original to this project. It uses made-up words only: no real brands, writers or game characters.

| Files | Source | Author | Licence |
|---|---|---|---|
| `viewer/public/models/skater_male.glb`, `skater_female.glb` | "Universal Base Characters", Standard (free) version: `Superhero_Male_FullBody`, `Superhero_Female_FullBody`. https://quaternius.itch.io/universal-base-characters | Quaternius (https://quaternius.com) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/), stated in the pack's `License_Standard.txt` |
| `viewer/public/models/skin_male.png`, `skin_female.png`, `eye.png` | Same pack: `T_Superhero_Male_Dark`, `T_Superhero_Female_Dark_BaseColor`, `T_Eye_Brown` | Quaternius | CC0 1.0 |

## What was changed

`viewer/scripts/prepare-models.mjs` strips the normal and roughness maps, extra UV sets and vertex colours, and
simplifies each model to about 8k triangles (14.3k -> 7.8k male, 15.1k -> 7.8k female). Textures are resized to 512 px and 128 px.
At load, `viewer/src/views/three/characters/street/` reshapes the mesh into clothes and paints an outfit over the skin texture.

The animations are the project's own code-driven poses (`characters/poses.ts`). No third-party animation clips are used.
