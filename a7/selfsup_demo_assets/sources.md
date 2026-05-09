# Self-Supervised Demo Image Sources

These images are prepared for a small web/app demo of self-supervised visual learning tasks, such as rotation prediction, masked image reconstruction, and data augmentation comparison.

All local images are resized to a maximum side length of 512px for lightweight browser use.

| Local file | Suggested label | Source page | License noted on source |
| --- | --- | --- | --- |
| `images/cat.jpg` | Cat | https://commons.wikimedia.org/wiki/File:Domestic_Cat.jpg | Public domain |
| `images/dog.jpg` | Dog | https://commons.wikimedia.org/wiki/File:Photo_of_a_dog.jpg | CC0 1.0 |
| `images/car.jpg` | Car | https://commons.wikimedia.org/wiki/File:Car_(23).jpg | Public domain |
| `images/flower.jpg` | Flower | https://commons.wikimedia.org/wiki/File:Dandelion_flower_photo.jpg | CC0 1.0 |
| `images/mountain.jpg` | Mountain | https://commons.wikimedia.org/wiki/File:Mountains_Landscape.jpg | CC0 1.0 |
| `images/building.jpg` | Building | https://commons.wikimedia.org/wiki/File:A_architecture.jpg | CC0 1.0 |
| `images/fruit.jpg` | Fruit | https://commons.wikimedia.org/wiki/File:Fruits_(1).jpg | Public domain |
| `images/bicycle.jpg` | Bicycle | https://commons.wikimedia.org/wiki/File:Bicycles.jpg | Public domain |

Recommended use in the app:

- Use all eight images for the image selector.
- Use `cat.jpg`, `car.jpg`, and `building.jpg` for rotation prediction, because their upright orientation is visually obvious.
- Use `flower.jpg`, `fruit.jpg`, and `mountain.jpg` for masked reconstruction, because color and texture changes are easy to see.
- Use `dog.jpg`, `bicycle.jpg`, and `building.jpg` for augmentation comparison, because crop, flip, blur, and color jitter produce visible differences.
