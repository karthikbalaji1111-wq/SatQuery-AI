"""Render the Earth still used as the landing page's non-WebGL fallback.

    /Applications/Blender.app/Contents/MacOS/Blender --background \
        --python tools/blender/render_earth_still.py

Why a rendered still rather than a Blender-exported Earth: the live web Earth is
a shader sphere whose day/night terminator is computed per frame from the same
sun vector that lights the satellite. Baking that to a GLB would throw the
terminator away and ship a heavier asset for a worse result. So Blender is used
for the thing it is genuinely better at - a properly lit, path-traced frame -
and that frame becomes the fallback for visitors with no WebGL, a weak GPU, or
reduced-motion preferences, replacing what was previously a CSS gradient.

Framing deliberately matches the web hero (planet right of centre, sun from the
upper left) so the fallback and the live scene are recognisably the same shot.
"""

import math
import os
import sys

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEX = os.path.join(ROOT, "frontend", "public", "landing")
OUT = os.path.join(TEX, "earth_still.png")


def clear() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def build() -> None:
    clear()
    scene = bpy.context.scene

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 1200
    # Transparent film: the page supplies its own black, and an alpha edge lets
    # the still sit on the same background the live canvas uses.
    scene.render.film_transparent = True

    world = bpy.data.worlds.new("Space")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.004, 0.006, 0.012, 1)

    # ---------------------------------------------------------------- Earth
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=128, ring_count=64)
    earth = bpy.context.object
    earth.name = "Earth"
    bpy.ops.object.shade_smooth()

    mat = bpy.data.materials.new("EarthSurface")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    day = nt.nodes.new("ShaderNodeTexImage")
    day.image = bpy.data.images.load(os.path.join(TEX, "earth_day.jpg"))
    day.location = (-700, 300)

    # The same lift the web shader applies, so the fallback and the live scene
    # agree about how bright the planet is.
    gamma = nt.nodes.new("ShaderNodeGamma")
    gamma.inputs["Gamma"].default_value = 0.82
    gamma.location = (-460, 300)
    nt.links.new(day.outputs["Color"], gamma.inputs["Color"])
    nt.links.new(gamma.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.6

    # City lights. On the sunlit half the diffuse term swamps this, so it reads
    # only where the surface is actually dark - no extra masking needed.
    night = nt.nodes.new("ShaderNodeTexImage")
    night.image = bpy.data.images.load(os.path.join(TEX, "earth_night.jpg"))
    night.location = (-700, -200)
    nt.links.new(night.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 0.42
    earth.data.materials.append(mat)

    # --------------------------------------------------------------- clouds
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.012, segments=96, ring_count=48)
    clouds = bpy.context.object
    clouds.name = "Clouds"
    bpy.ops.object.shade_smooth()

    cmat = bpy.data.materials.new("Clouds")
    cmat.use_nodes = True
    cnt = cmat.node_tree
    cnt.nodes.remove(cnt.nodes["Principled BSDF"])
    cout = cnt.nodes["Material Output"]
    ctex = cnt.nodes.new("ShaderNodeTexImage")
    ctex.image = bpy.data.images.load(os.path.join(TEX, "earth_clouds.jpg"))
    ctex.location = (-700, 0)
    ramp = cnt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.32
    ramp.color_ramp.elements[1].position = 0.9
    ramp.location = (-460, 0)
    cnt.links.new(ctex.outputs["Color"], ramp.inputs["Fac"])
    diffuse = cnt.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Roughness"].default_value = 1.0
    diffuse.location = (-240, 120)
    transparent = cnt.nodes.new("ShaderNodeBsdfTransparent")
    transparent.location = (-240, -80)
    mix = cnt.nodes.new("ShaderNodeMixShader")
    mix.location = (-40, 0)
    cnt.links.new(ramp.outputs["Color"], mix.inputs["Fac"])
    cnt.links.new(transparent.outputs[0], mix.inputs[1])
    cnt.links.new(diffuse.outputs[0], mix.inputs[2])
    cnt.links.new(mix.outputs[0], cout.inputs["Surface"])
    clouds.data.materials.append(cmat)

    # ----------------------------------------------------------- atmosphere
    # Fresnel-gated shell: visible only at grazing angles, so it reads as a
    # limb rather than as an outline drawn round a circle.
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.03, segments=96, ring_count=48)
    atmo = bpy.context.object
    atmo.name = "Atmosphere"
    bpy.ops.object.shade_smooth()
    amat = bpy.data.materials.new("Atmosphere")
    amat.use_nodes = True
    ant = amat.node_tree
    ant.nodes.remove(ant.nodes["Principled BSDF"])
    aout = ant.nodes["Material Output"]
    fres = ant.nodes.new("ShaderNodeFresnel")
    fres.inputs["IOR"].default_value = 1.28
    fres.location = (-600, 0)
    power = ant.nodes.new("ShaderNodeMath")
    power.operation = "POWER"
    power.inputs[1].default_value = 4.6
    power.location = (-420, 0)
    ant.links.new(fres.outputs["Fac"], power.inputs[0])

    # Gate by sunlight: dot(normal, sun) via a Geometry node, mapped so the
    # limb glows only where the sun actually reaches it.
    geo = ant.nodes.new("ShaderNodeNewGeometry")
    geo.location = (-820, -260)
    sun_vec = ant.nodes.new("ShaderNodeVectorMath")
    sun_vec.operation = "DOT_PRODUCT"
    sun_vec.inputs[1].default_value = (-0.62, -0.55, 0.56)
    sun_vec.location = (-600, -260)
    ant.links.new(geo.outputs["Normal"], sun_vec.inputs[0])
    gate = ant.nodes.new("ShaderNodeMapRange")
    gate.inputs["From Min"].default_value = -0.35
    gate.inputs["From Max"].default_value = 0.30
    gate.location = (-420, -260)
    ant.links.new(sun_vec.outputs["Value"], gate.inputs["Value"])
    combine = ant.nodes.new("ShaderNodeMath")
    combine.operation = "MULTIPLY"
    combine.location = (-240, -160)
    ant.links.new(power.outputs["Value"], combine.inputs[0])
    ant.links.new(gate.outputs["Result"], combine.inputs[1])
    emit = ant.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (0.32, 0.52, 0.88, 1)
    emit.inputs["Strength"].default_value = 0.85
    emit.location = (-240, 120)
    atrans = ant.nodes.new("ShaderNodeBsdfTransparent")
    atrans.location = (-240, -100)
    amix = ant.nodes.new("ShaderNodeMixShader")
    amix.location = (-40, 0)
    ant.links.new(combine.outputs["Value"], amix.inputs["Fac"])
    ant.links.new(atrans.outputs[0], amix.inputs[1])
    ant.links.new(emit.outputs[0], amix.inputs[2])
    ant.links.new(amix.outputs[0], aout.inputs["Surface"])
    atmo.data.materials.append(amat)

    # ------------------------------------------------------------------ sun
    bpy.ops.object.light_add(type="SUN", location=(5, -4, 3))
    sun = bpy.context.object
    sun.data.energy = 4.0
    # A real angular diameter gives a terminator of the right softness for free.
    sun.data.angle = math.radians(0.53)
    sun.data.color = (1.0, 0.96, 0.90)
    sun.rotation_euler = (math.radians(64), 0, math.radians(-42))

    # --------------------------------------------------------------- camera
    bpy.ops.object.camera_add(location=(0, -4.15, 0.58))
    cam = bpy.context.object
    scene.camera = cam
    cam.data.lens = 62
    cam.rotation_euler = (math.radians(82), 0, 0)

    print(f"EARTH_SCENE {[o.name for o in bpy.data.objects]}")


def render() -> None:
    bpy.context.scene.render.filepath = OUT
    bpy.ops.render.render(write_still=True)
    print(f"EARTH_STILL {OUT} {os.path.getsize(OUT)}")


build()
render()
print("RENDER_DONE")
sys.exit(0)
