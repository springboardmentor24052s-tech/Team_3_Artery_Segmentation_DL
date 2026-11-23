
import numpy as np
from scipy import ndimage
from app.stenosis_classification import classify_severity

def _estimate_width(image, center, normal, search_radius=8):
    """Estimate dark-vessel width across a perpendicular image profile."""

    offsets = np.arange(-25, 26, 0.5)
    shifts = np.arange(-search_radius, search_radius + 1, 1)

    profiles = []

    for shift in shifts:
        cx = center[0] + shift * (-normal[1])
        cy = center[1] + shift * normal[0]

        x = cx + offsets * normal[0]
        y = cy + offsets * normal[1]

        profile = ndimage.map_coordinates(
            image,
            [y, x],
            order=1,
            mode="nearest"
        )
        profiles.append(profile)

    profiles = np.asarray(profiles)

    # Select the darkest cross-sectional profile near the estimated centre.
    darkness = profiles[:, 40:61].mean(axis=1)
    profile = profiles[np.argmin(darkness)]

    background = np.median(
        np.concatenate([profile[:12], profile[-12:]])
    )
    minimum = float(profile.min())
    contrast = background - minimum

    if contrast < 0.04:
        return None

    threshold = background - 0.5 * contrast
    darkest_index = int(np.argmin(profile))

    left = darkest_index
    right = darkest_index

    while left > 0 and profile[left - 1] < threshold:
        left -= 1

    while right < len(profile) - 1 and profile[right + 1] < threshold:
        right += 1

    width = (right - left + 1) * 0.5

    if width < 1.0 or width > 30:
        return None

    return {
        "width": float(width),
        "contrast": float(contrast)
    }


def measure_stenosis(image, binary_mask):
    """Experimental image-based stenosis diameter estimation."""

    image = np.asarray(image, dtype=np.float32)
    binary_mask = np.asarray(binary_mask) > 0

    if image.shape != binary_mask.shape:
        raise ValueError("Image and stenosis mask dimensions must match.")

    if not np.isfinite(image).all():
        raise ValueError("Image contains invalid values.")

    labels, count = ndimage.label(binary_mask)
    measurements = []

    for region_id in range(1, count + 1):
        region = labels == region_id
        y, x = np.where(region)

        result = {
            "region_id": region_id,
            "measurement_available": False,
            "minimum_diameter_pixels": None,
            "reference_diameter_pixels": None,
            "percentage_narrowing": None,
            "severity": "Unavailable",
            "measurement_type": "Experimental image-based estimate"
        }

        if len(x) < 10:
            measurements.append(result)
            continue

        points = np.column_stack((x, y)).astype(np.float32)
        center = points.mean(axis=0)

        covariance = np.cov(points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        if eigenvalues[0] <= 0:
            measurements.append(result)
            continue

        # Require a sufficiently elongated lesion region to estimate direction.
        if eigenvalues[1] / eigenvalues[0] < 1.2:
            measurements.append(result)
            continue

        axis = eigenvectors[:, 1]
        normal = np.array([-axis[1], axis[0]])

        projected = (points - center) @ axis
        lesion_half_length = float(
            np.percentile(np.abs(projected), 75)
        )

        if lesion_half_length < 2:
            measurements.append(result)
            continue

        # Estimate minimum width around the lesion.
        lesion_widths = []

        for position in [-3, 0, 3]:
            sample_center = center + position * axis
            measurement = _estimate_width(
                image, sample_center, normal
            )

            if measurement:
                lesion_widths.append(measurement["width"])

        # Estimate reference widths beyond the lesion.
        reference_widths = []

        for distance in [
            lesion_half_length + 10,
            lesion_half_length + 17,
            lesion_half_length + 24
        ]:
            for direction in [-1, 1]:
                sample_center = center + direction * distance * axis
                measurement = _estimate_width(
                    image, sample_center, normal
                )

                if measurement:
                    reference_widths.append(measurement["width"])

        if not lesion_widths or len(reference_widths) < 2:
            measurements.append(result)
            continue

        minimum_diameter = min(lesion_widths)
        reference_diameter = float(np.median(reference_widths))

        if minimum_diameter >= reference_diameter:
            measurements.append(result)
            continue

        percentage = (
            (reference_diameter - minimum_diameter)
            / reference_diameter
        ) * 100

        result.update({
    "measurement_available": True,
    "minimum_diameter_pixels": round(minimum_diameter, 2),
    "reference_diameter_pixels": round(reference_diameter, 2),
    "percentage_narrowing": round(percentage, 2),
    "severity": classify_severity(round(percentage, 2))
})

        measurements.append(result)

    return {
        "measurement_method": "Cross-sectional angiogram intensity profiles",
        "region_count": count,
        "successful_measurements": sum(
            item["measurement_available"] for item in measurements
        ),
        "measurements": measurements,
        "note": (
            "Experimental estimates only. Not clinically validated."
        )
    }
