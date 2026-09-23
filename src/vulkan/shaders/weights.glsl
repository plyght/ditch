// Weight loads shared by the matrix kernels. A weight tile arrives as the raw
// bytes of its on-disk dtype, bound as an array of 32-bit words, and every
// element is converted as it is read:
//
//   DTYPE 0  f32   one word per element
//   DTYPE 1  f16   two elements per word, unpackHalf2x16 (core GLSL 4.20:
//                  needs neither shaderFloat16 nor 16-bit storage)
//   DTYPE 2  bf16  two elements per word, shifted into the f32 exponent field
//
// Element 0 of a pair is the low half of the word (little-endian storage), so
// the byte layout on the device is exactly the one on disk.

float load_w(uint i) {
#if DTYPE == 0
    return uintBitsToFloat(w[i]);
#elif DTYPE == 1
    const vec2 pair = unpackHalf2x16(w[i >> 1]);
    return (i & 1u) == 0u ? pair.x : pair.y;
#else
    const uint word = w[i >> 1];
    return uintBitsToFloat((i & 1u) == 0u ? (word << 16) : (word & 0xffff0000u));
#endif
}
