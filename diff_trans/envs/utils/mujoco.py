from mujoco import mjx


# Function to get geometry name
def get_geom_name(model, geom_id):
    # Ensure the geometry ID is within valid range
    if geom_id < 0 or geom_id >= model.ngeom:
        return None  # Invalid geometry ID

    # Get the starting index of the geometry name
    name_index = model.name_geomadr[geom_id]

    # Return the geometry name
    return model.names[name_index:].decode().split('\x00', 1)[0]


# Function to get body name
def get_body_name(model: mjx.Model, body_id: int):
    # Ensure the geometry ID is within valid range
    if body_id < 0 or body_id >= model.nbody:
        return None  # Invalid body ID

    # Get the starting index of the geometry name
    name_index = model.name_bodyadr[body_id]

    # Return the geometry name
    return model.names[name_index:].decode().split('\x00', 1)[0]