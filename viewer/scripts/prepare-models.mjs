// Turns the Quaternius "Universal Base Characters [Standard]" glTFs (CC0, see ../../ASSETS.md) into the two
// small GLBs the mid-2000s look uses (viewer/public/models/). Run once; the outputs are committed.
//
//   node scripts/prepare-models.mjs "<unzipped>/Universal Base Characters[Standard]"
//
// Keeps: skeleton, position, normal, first UV set, joints, weights. Base-colour + eye textures are written
// next to the GLB, capped at 512 px (with macOS `sips`); the runtime paints outfits over them.
// Drops: every material texture from the GLB, extra UV sets, vertex colours. Then simplifies (meshoptimizer, same
// algorithm as `npx @gltf-transform/cli simplify`) so body + eyes + brows land at about 8k triangles.
import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { NodeIO } from '@gltf-transform/core'
import { prune, simplify, weld } from '@gltf-transform/functions'
import { MeshoptSimplifier } from 'meshoptimizer'

const src = process.argv[2]
if (!src) throw new Error('usage: prepare-models.mjs <unzipped pack dir>')
const OUT = resolve(dirname(fileURLToPath(import.meta.url)), '../public/models')
mkdirSync(OUT, { recursive: true })
const TARGET_TRIS = 8000
const KEEP = new Set(['POSITION', 'NORMAL', 'TEXCOORD_0', 'JOINTS_0', 'WEIGHTS_0'])

const countTris = doc => doc.getRoot().listMeshes().flatMap(m => m.listPrimitives())
  .reduce((n, p) => n + (p.getIndices()?.getCount() ?? 0) / 3, 0)

await MeshoptSimplifier.ready
const io = new NodeIO()
const DIR = `${src}/Base Characters/Godot - UE`

/** The pack's .gltf files name a few images that aren't in the zip, so strip images before reading. */
async function readWithoutImages(file) {
  const json = JSON.parse(readFileSync(`${DIR}/${file}.gltf`, 'utf8'))
  for (const m of json.materials ?? []) {
    delete m.normalTexture; delete m.occlusionTexture
    if (m.pbrMetallicRoughness) { delete m.pbrMetallicRoughness.baseColorTexture; delete m.pbrMetallicRoughness.metallicRoughnessTexture }
  }
  delete json.images; delete json.textures; delete json.samplers
  const resources = Object.fromEntries(json.buffers.map(b => [b.uri, new Uint8Array(readFileSync(`${DIR}/${b.uri}`))]))
  return io.readJSON({ json, resources })
}

function capTexture(from, to, px) {
  if (!existsSync(`${DIR}/${from}`)) throw new Error(`missing ${from}`)
  execFileSync('sips', ['-Z', String(px), `${DIR}/${from}`, '--out', `${OUT}/${to}`], { stdio: 'ignore' })
}
capTexture('T_Superhero_Male_Dark.png', 'skin_male.png', 512)
capTexture('T_Superhero_Female_Dark_BaseColor.png', 'skin_female.png', 512)
capTexture('T_Eye_Brown.png', 'eye.png', 128)

for (const [name, file] of [['skater_male', 'Superhero_Male_FullBody'], ['skater_female', 'Superhero_Female_FullBody']]) {
  const doc = await readWithoutImages(file)
  for (const mesh of doc.getRoot().listMeshes()) for (const prim of mesh.listPrimitives()) {
    for (const sem of prim.listSemantics()) if (!KEEP.has(sem)) prim.setAttribute(sem, null)
  }
  const before = countTris(doc)
  await doc.transform(prune({ keepAttributes: true }), weld(), simplify({ simplifier: MeshoptSimplifier, ratio: (TARGET_TRIS / before) * 0.98, error: 0.01, lockBorder: true }), prune({ keepAttributes: true }))
  await io.write(`${OUT}/${name}.glb`, doc)
  console.log(name, before, '->', countTris(doc), 'tris')
}
