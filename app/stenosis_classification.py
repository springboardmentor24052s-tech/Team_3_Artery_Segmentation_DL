def calculate_percentage_narrowing(minimum_diameter, reference_diameter):
    if minimum_diameter is None or reference_diameter is None:
        return None

    if reference_diameter <= 0 or minimum_diameter < 0:
        raise ValueError("Diameters must be non-negative and reference diameter must be positive.")

    if minimum_diameter > reference_diameter:
        raise ValueError("Minimum diameter cannot exceed reference diameter.")

    percentage = (
        (reference_diameter - minimum_diameter)
        / reference_diameter
    ) * 100

    return round(percentage, 2)


def classify_severity(percentage_narrowing):
    if percentage_narrowing is None:
        return "Unavailable"

    if not 0 <= percentage_narrowing <= 100:
        raise ValueError("Percentage narrowing must be between 0 and 100.")

    if percentage_narrowing < 50:
        return "Mild"
    elif percentage_narrowing < 70:
        return "Moderate"
    else:
        return "Severe"


def analyze_stenosis(minimum_diameter, reference_diameter):
    percentage = calculate_percentage_narrowing(
        minimum_diameter,
        reference_diameter
    )

    return {
        "minimum_diameter_pixels": minimum_diameter,
        "reference_diameter_pixels": reference_diameter,
        "percentage_narrowing": percentage,
        "severity": classify_severity(percentage),
        "measurement_available": percentage is not None
    }
