// ---------------------------------------------------------------------------
// Bakes the static part of the scene into a 3D distance texture: one compute
// thread per voxel, voxel centres spanning `frame.boundsMin .. + boundsSize`.
// Requires (prepended): sdf.wgsl, the `Frame` struct, scene.wgsl.
// ---------------------------------------------------------------------------

@group(0) @binding(1) var baked_out: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn cs_bake(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(baked_out);
  if (any(id >= dims)) { return; }
  let p = frame.boundsMin + (vec3f(id) + 0.5) / vec3f(dims) * frame.boundsSize;
  textureStore(baked_out, id, vec4f(map_static(p).x, 0.0, 0.0, 1.0));
}
