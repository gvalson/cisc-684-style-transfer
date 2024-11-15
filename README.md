# Referencing an Artist's Overall Style in Style Transfer



## Initial Attempt

We used an average Gram matrix of all the images of one artist in a specific style. For example, we averaged Pablo Picasso's Cubist paintings and then applied the style to our original image. The same process was repeated for the other artists.

## Trying to Separate Color from "Texture"

We first convert the image from RGB/BGR space into LAB space, which represents the luminosity or lightness of each pixel separately from its color. This allows us to directly modify the value (i.e lightness or darkness) of each pixel in our source image without changing its color.
