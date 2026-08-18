# Gestures 3D Navigation for Blender

This folder contains the Blender side of the local two-hand navigation
integration. The add-on listens on `127.0.0.1:8765` and applies continuous
`orbit_x`, `orbit_y`, `pan_x`, `pan_y`, `zoom`, and `roll` values on Blender's
main thread. It never uses simulated keyboard or mouse input.

## Install

1. Zip the `gestures_navigation` folder so `gestures_navigation/__init__.py` is
   at the top level of the archive.
2. In Blender, open **Edit → Preferences → Add-ons → Install…**, select the zip,
   and enable **Gestures 3D Navigation**.
3. In a 3D Viewport press **N**, open the **Gestures** tab, and enable
   navigation.
4. Start the Gestures desktop application and enable its 3D Navigation panel.

The application sends only localhost UDP packets. The receiver port can be
changed in the panel; restart the receiver after changing it. The application's
reply port defaults to `8766`.

## Targets

- **Viewport** directly changes visible `RegionView3D` rotation, location, and
  view distance.
- **Camera** changes the selected camera's actual transform, falling back to
  the active scene camera.

The Blender sidebar contains independent sensitivity multipliers and a
**Follow Gestures mode** option. Keep the add-on's **Enable Navigation** switch
off until the application is connected and the hand activation pose is ready.
