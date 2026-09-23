-- Helpers every model definition sees (docs/models.md). Natives from Zig:
-- warn(msg), unsupported(msg), invalid(msg), require(lib) (a library of
-- src/models/lib), rope_scaling(rs, cfg, rotary_dim, max_positions) (ditch's
-- reading of a rope_scaling table), yarn_mscale(scale, mscale), log32(x)
-- (single-precision log) and pow(x, y) (ditch's double-precision power). A definition's
-- `config` function receives the model's config.json as `cfg` (JSON objects
-- and arrays are tables, arrays 1-based, a JSON null is `null`) and the
-- parsed configuration as `c`, which it edits in place.

-- A number, or `d` when `v` is missing, null or not a number.
function num(v, d)
  if type(v) == "number" then return v end
  return d
end

-- A non-negative integer (a float is truncated), or `d` when `v` is missing,
-- null, negative or not a number: how ditch reads every integer key.
function int(v, d)
  if type(v) ~= "number" or v < 0 then return d end
  return math.tointeger(math.floor(v))
end

-- A boolean (an integer counts as true when non-zero), or `d`.
function flag(v, d)
  if type(v) == "boolean" then return v end
  if math.type(v) == "integer" then return v ~= 0 end
  return d
end

-- A string, or nil.
function str(v)
  if type(v) == "string" then return v end
  return nil
end

-- A JSON object (a table that is not `null` nor a non-empty array), or nil.
-- An empty array cannot be told from an empty object and counts as one.
function obj(v)
  if type(v) == "table" and v ~= null and #v == 0 then return v end
  return nil
end

-- Whether a key is present at all (a JSON null counts as present).
function present(v)
  return v ~= nil
end

-- `x` rounded to single precision: ditch keeps its scalars as f32, so a
-- computation that must match it bit for bit rounds after every operation.
function f32(x)
  return (string.unpack("f", string.pack("f", x)))
end

-- Sets `list[i + 1] = f(i)` for every zero-based layer index `i` of `list`
-- (a per-layer table of `c`, e.g. `c.sliding_layers`).
function each_layer(list, f)
  for i = 0, #list - 1 do list[i + 1] = f(i) end
  return list
end

-- The number of elements of a JSON array (0 for anything else).
function len(v)
  if type(v) == "table" and v ~= null then return #v end
  return 0
end
