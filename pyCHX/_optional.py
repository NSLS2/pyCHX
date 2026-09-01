"""Small fallbacks for dependencies that are optional in source installs."""

try:
    from modest_image import imshow
except ImportError:

    def imshow(axes, image, *args, **kwargs):
        """Fall back to Matplotlib when ModestImage is unavailable."""
        return axes.imshow(image, *args, **kwargs)


__all__ = ["imshow"]
