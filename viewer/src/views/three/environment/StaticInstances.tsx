// Draw many copies of one mesh in a single draw call. Transforms are set once (static props).
import { useLayoutEffect, useRef, type ReactNode } from 'react'
import { Color, Euler, InstancedMesh, Matrix4, Quaternion, Vector3 } from 'three'

export interface InstanceXform {
  pos: [number, number, number]
  rot?: [number, number, number]
  scale?: [number, number, number]
  color?: string
}

export function StaticInstances({ items, children, castShadow, receiveShadow }: {
  items: InstanceXform[]; children: ReactNode; castShadow?: boolean; receiveShadow?: boolean
}) {
  const ref = useRef<InstancedMesh>(null)
  useLayoutEffect(() => {
    const m = ref.current; if (!m) return
    const mat = new Matrix4(), q = new Quaternion(), e = new Euler(), p = new Vector3(), s = new Vector3(), c = new Color()
    items.forEach((it, i) => {
      e.set(...(it.rot ?? [0, 0, 0])); q.setFromEuler(e)
      p.set(...it.pos); s.set(...(it.scale ?? [1, 1, 1]))
      m.setMatrixAt(i, mat.compose(p, q, s))
      if (it.color) m.setColorAt(i, c.set(it.color))
    })
    m.count = items.length
    m.instanceMatrix.needsUpdate = true
    if (m.instanceColor) m.instanceColor.needsUpdate = true
    m.computeBoundingSphere()
  }, [items])
  return (
    <instancedMesh ref={ref} args={[undefined, undefined, items.length]} castShadow={castShadow} receiveShadow={receiveShadow}>
      {children}
    </instancedMesh>
  )
}
