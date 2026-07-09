/**
 * Parse NumPy .npy binary format into Float32Array + shape.
 * Supports format versions 1.0 and 2.0, float32 ('<f4') only.
 */
export function parseNpy(buffer) {
  const view = new DataView(buffer);
  const major = view.getUint8(6);
  const headerLen = major === 1
    ? view.getUint16(8, true)
    : view.getUint32(8, true);
  const headerOffset = major === 1 ? 10 : 12;
  const headerStr = new TextDecoder().decode(
    new Uint8Array(buffer, headerOffset, headerLen)
  );
  const shapeMatch = headerStr.match(/shape['"]\s*:\s*\(([^)]+)\)/);
  if (!shapeMatch) throw new Error('Cannot parse npy shape from header');
  const shape = shapeMatch[1]
    .split(',')
    .map(s => parseInt(s.trim(), 10))
    .filter(n => !isNaN(n));
  const dataOffset = headerOffset + headerLen;
  const data = new Float32Array(buffer, dataOffset);
  return { data, shape };
}
