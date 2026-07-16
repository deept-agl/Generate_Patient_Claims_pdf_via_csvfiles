from __future__ import annotations

import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ============================================================
# PATH AND GENERATION CONFIGURATION
# ============================================================

# Folder containing generate_claims_document.py
BASE_DIR = Path(__file__).resolve().parent

# CSV files are inside:
# sample_healthcare_dataset/Synthea_source_dataset/
INPUT_DIR = BASE_DIR / "Synthea_source_dataset"

# Generated documents will be created inside:
# sample_healthcare_dataset/generated_healthcare_claims/
OUTPUT_DIR = BASE_DIR / "generated_healthcare_claims"

# Number of patients for which claim packages will be generated.
# Change to None later to process all patients.
MAX_PATIENTS: int | None = 10

RANDOM_SEED = 42

GENERATE_TEST_SCENARIOS = True

TEST_SCENARIOS = [
    "VALID_CLAIM",
    "UNPRESCRIBED_MEDICINE",
    "LOW_QUALITY_HANDWRITING",
    "MISSING_DISCHARGE_SUMMARY",
    "CLAIM_AMOUNT_MISMATCH",
]

random.seed(RANDOM_SEED)


# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip().upper() for column in df.columns]
    return df


def read_csv_file(filename: str, required: bool = False) -> pd.DataFrame:
    path = INPUT_DIR / filename

    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")

        print(f"Warning: {filename} was not found. Continuing without it.")
        return pd.DataFrame()

    dataframe = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    dataframe = clean_column_names(dataframe)

    print(f"Loaded {filename}: {len(dataframe):,} rows")
    return dataframe


def get_value(
    row: pd.Series | dict[str, Any] | None,
    *column_names: str,
    default: str = "",
) -> str:
    if row is None:
        return default

    for column_name in column_names:
        key = column_name.upper()

        if isinstance(row, pd.Series):
            if key in row.index:
                value = row[key]
            else:
                continue
        else:
            value = row.get(key)

        if value is not None and str(value).strip():
            return str(value).strip()

    return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        cleaned = re.sub(r"[^0-9.\-]", "", str(value))
        return float(cleaned) if cleaned else default
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def parse_date(value: Any) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None

    result = pd.to_datetime(value, errors="coerce", utc=True)

    if pd.isna(result):
        return None

    return result


def format_date(value: Any) -> str:
    parsed = parse_date(value)

    if parsed is None:
        return ""

    return parsed.strftime("%d-%m-%Y")


def format_datetime(value: Any) -> str:
    parsed = parse_date(value)

    if parsed is None:
        return ""

    return parsed.strftime("%d-%m-%Y %H:%M")


def sanitize_filename(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*]', "", value)
    value = re.sub(r"\s+", "_", value.strip())
    return value[:100] or "UNKNOWN_PATIENT"


def calculate_age(birthdate: str, reference_date: str) -> int | None:
    birth = parse_date(birthdate)
    reference = parse_date(reference_date)

    if birth is None or reference is None:
        return None

    age = reference.year - birth.year

    if (reference.month, reference.day) < (birth.month, birth.day):
        age -= 1

    return max(age, 0)


def first_row(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None

    return df.iloc[0]


def rows_for_patient(
    df: pd.DataFrame,
    patient_id: str,
    encounter_id: str | None = None,
) -> pd.DataFrame:
    if df.empty or "PATIENT" not in df.columns:
        return pd.DataFrame()

    result = df[df["PATIENT"].astype(str) == str(patient_id)].copy()

    if (
        encounter_id
        and "ENCOUNTER" in result.columns
        and result["ENCOUNTER"].astype(str).eq(str(encounter_id)).any()
    ):
        encounter_result = result[
            result["ENCOUNTER"].astype(str) == str(encounter_id)
        ].copy()

        if not encounter_result.empty:
            return encounter_result

    return result


# ============================================================
# PDF HELPERS
# ============================================================

STYLES = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle(
    "CustomTitle",
    parent=STYLES["Title"],
    fontName="Helvetica-Bold",
    fontSize=18,
    leading=22,
    alignment=TA_CENTER,
    spaceAfter=12,
)

SUBTITLE_STYLE = ParagraphStyle(
    "CustomSubtitle",
    parent=STYLES["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=12,
    leading=15,
    alignment=TA_LEFT,
    spaceBefore=8,
    spaceAfter=7,
)

BODY_STYLE = ParagraphStyle(
    "CustomBody",
    parent=STYLES["BodyText"],
    fontName="Helvetica",
    fontSize=9,
    leading=13,
)

SMALL_STYLE = ParagraphStyle(
    "Small",
    parent=STYLES["BodyText"],
    fontName="Helvetica",
    fontSize=8,
    leading=11,
)

RIGHT_STYLE = ParagraphStyle(
    "Right",
    parent=BODY_STYLE,
    alignment=TA_RIGHT,
)

CENTER_STYLE = ParagraphStyle(
    "Center",
    parent=BODY_STYLE,
    alignment=TA_CENTER,
)


def make_pdf(
    output_path: Path,
    story: list[Any],
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=title,
        author="Synthetic Healthcare Claims Generator",
    )

    document.build(story)


def information_table(rows: list[tuple[str, Any]]) -> Table:
    data = []

    for label, value in rows:
        data.append(
            [
                Paragraph(f"<b>{label}</b>", BODY_STYLE),
                Paragraph(str(value or "Not available"), BODY_STYLE),
            ]
        )

    table = Table(data, colWidths=[48 * mm, 110 * mm])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    return table


def line_item_table(
    headers: list[str],
    rows: list[list[Any]],
    widths: list[float],
) -> Table:
    data = [
        [Paragraph(f"<b>{header}</b>", SMALL_STYLE) for header in headers]
    ]

    for row in rows:
        data.append(
            [Paragraph(str(value), SMALL_STYLE) for value in row]
        )

    table = Table(data, colWidths=widths, repeatRows=1)

    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D6EAF8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    return table


# ============================================================
# CLAIM DATA PREPARATION
# ============================================================

def load_synthea_data() -> dict[str, pd.DataFrame]:
    return {
        "patients": read_csv_file("patients.csv", required=True),
        "encounters": read_csv_file("encounters.csv", required=True),
        "conditions": read_csv_file("conditions.csv"),
        "medications": read_csv_file("medications.csv"),
        "procedures": read_csv_file("procedures.csv"),
        "observations": read_csv_file("observations.csv"),
        "organizations": read_csv_file("organizations.csv"),
        "providers": read_csv_file("providers.csv"),
        "payers": read_csv_file("payers.csv"),
        "imaging_studies": read_csv_file("imaging_studies.csv"),
        "allergies": read_csv_file("allergies.csv"),
        "careplans": read_csv_file("careplans.csv"),
    }


def choose_encounter(
    patient_id: str,
    encounters: pd.DataFrame,
) -> pd.Series | None:
    if encounters.empty or "PATIENT" not in encounters.columns:
        return None

    patient_encounters = encounters[
        encounters["PATIENT"].astype(str) == str(patient_id)
    ].copy()

    if patient_encounters.empty:
        return None

    if "START" in patient_encounters.columns:
        patient_encounters["_START_SORT"] = pd.to_datetime(
            patient_encounters["START"],
            errors="coerce",
            utc=True,
        )

        patient_encounters = patient_encounters.sort_values(
            "_START_SORT",
            ascending=False,
        )

    return patient_encounters.iloc[0]


def find_organization(
    organization_id: str,
    organizations: pd.DataFrame,
) -> pd.Series | None:
    if (
        organizations.empty
        or "ID" not in organizations.columns
        or not organization_id
    ):
        return None

    match = organizations[
        organizations["ID"].astype(str) == str(organization_id)
    ]

    return first_row(match)


def find_provider(
    provider_id: str,
    providers: pd.DataFrame,
) -> pd.Series | None:
    if providers.empty or "ID" not in providers.columns or not provider_id:
        return None

    match = providers[
        providers["ID"].astype(str) == str(provider_id)
    ]

    return first_row(match)


def find_payer(
    payer_id: str,
    payers: pd.DataFrame,
) -> pd.Series | None:
    if payers.empty or "ID" not in payers.columns or not payer_id:
        return None

    match = payers[
        payers["ID"].astype(str) == str(payer_id)
    ]

    return first_row(match)


def build_patient_name(patient: pd.Series) -> str:
    first = get_value(patient, "FIRST", default="Unknown")
    middle = get_value(patient, "MIDDLE")
    last = get_value(patient, "LAST", default="Patient")

    return " ".join(
        value for value in [first, middle, last] if value
    ).strip()


def prepare_medications(
    medication_rows: pd.DataFrame,
) -> list[dict[str, Any]]:
    medicines: list[dict[str, Any]] = []

    for _, row in medication_rows.head(8).iterrows():
        name = get_value(
            row,
            "DESCRIPTION",
            default="Prescribed medication",
        )

        base_cost = safe_float(
            get_value(row, "BASE_COST"),
            default=random.uniform(120, 600),
        )

        total_cost = safe_float(
            get_value(row, "TOTALCOST", "TOTAL_COST"),
            default=base_cost,
        )

        dispenses = max(
            safe_int(get_value(row, "DISPENSES"), default=1),
            1,
        )

        quantity = max(
            int(round(total_cost / base_cost)) if base_cost > 0 else dispenses,
            1,
        )

        medicines.append(
            {
                "name": name,
                "strength": random.choice(
                    ["250 mg", "500 mg", "40 mg", "10 mg", "5 ml"]
                ),
                "dosage": random.choice(
                    ["1 tablet", "1 capsule", "10 ml", "As directed"]
                ),
                "frequency": random.choice(
                    [
                        "Once daily",
                        "Twice daily",
                        "Three times daily",
                        "At bedtime",
                    ]
                ),
                "duration": random.choice(
                    ["5 days", "7 days", "10 days", "14 days"]
                ),
                "instructions": random.choice(
                    [
                        "After meals",
                        "Before breakfast",
                        "With water",
                        "As advised",
                    ]
                ),
                "quantity": quantity,
                "unit_price": round(base_cost, 2),
                "total": round(base_cost * quantity, 2),
            }
        )

    if not medicines:
        medicines = [
            {
                "name": "Paracetamol",
                "strength": "500 mg",
                "dosage": "1 tablet",
                "frequency": "Twice daily",
                "duration": "5 days",
                "instructions": "After meals",
                "quantity": 10,
                "unit_price": 8.0,
                "total": 80.0,
            }
        ]

    return medicines


def prepare_procedures(
    procedure_rows: pd.DataFrame,
    encounter_base_cost: float,
) -> list[dict[str, Any]]:
    procedures: list[dict[str, Any]] = []

    for _, row in procedure_rows.head(10).iterrows():
        description = get_value(
            row,
            "DESCRIPTION",
            default="Medical procedure",
        )

        amount = safe_float(
            get_value(row, "BASE_COST"),
            default=random.uniform(700, 4500),
        )

        procedures.append(
            {
                "description": description,
                "quantity": 1,
                "rate": round(amount, 2),
                "amount": round(amount, 2),
            }
        )

    if not procedures:
        procedures = [
            {
                "description": "Medical consultation",
                "quantity": 1,
                "rate": round(max(encounter_base_cost, 850), 2),
                "amount": round(max(encounter_base_cost, 850), 2),
            }
        ]

    return procedures


def prepare_observations(
    observation_rows: pd.DataFrame,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []

    for _, row in observation_rows.head(12).iterrows():
        observations.append(
            {
                "date": format_date(get_value(row, "DATE")),
                "test_name": get_value(
                    row,
                    "DESCRIPTION",
                    default="Clinical observation",
                ),
                "value": get_value(row, "VALUE", default="Not reported"),
                "units": get_value(row, "UNITS"),
                "category": get_value(row, "CATEGORY"),
            }
        )

    if not observations:
        observations = [
            {
                "date": datetime.now().strftime("%d-%m-%Y"),
                "test_name": "General clinical assessment",
                "value": "Within expected range",
                "units": "",
                "category": "Clinical",
            }
        ]

    return observations


def build_claim_record(
    claim_number: int,
    patient: pd.Series,
    encounter: pd.Series,
    data: dict[str, pd.DataFrame],
    scenario: str,
) -> dict[str, Any]:
    patient_id = get_value(patient, "ID")
    encounter_id = get_value(encounter, "ID")

    patient_name = build_patient_name(patient)

    organization_id = get_value(encounter, "ORGANIZATION")
    provider_id = get_value(encounter, "PROVIDER")
    payer_id = get_value(encounter, "PAYER")

    organization = find_organization(
        organization_id,
        data["organizations"],
    )

    provider = find_provider(
        provider_id,
        data["providers"],
    )

    payer = find_payer(
        payer_id,
        data["payers"],
    )

    condition_rows = rows_for_patient(
        data["conditions"],
        patient_id,
        encounter_id,
    )

    medication_rows = rows_for_patient(
        data["medications"],
        patient_id,
        encounter_id,
    )

    procedure_rows = rows_for_patient(
        data["procedures"],
        patient_id,
        encounter_id,
    )

    observation_rows = rows_for_patient(
        data["observations"],
        patient_id,
        encounter_id,
    )

    encounter_start = get_value(encounter, "START")
    encounter_stop = get_value(encounter, "STOP", default=encounter_start)

    age = calculate_age(
        get_value(patient, "BIRTHDATE"),
        encounter_start,
    )

    primary_condition = first_row(condition_rows)

    diagnosis = get_value(
        primary_condition,
        "DESCRIPTION",
        default=get_value(
            encounter,
            "REASONDESCRIPTION",
            "DESCRIPTION",
            default="General medical treatment",
        ),
    )

    encounter_base_cost = safe_float(
        get_value(encounter, "BASE_ENCOUNTER_COST"),
        default=1000,
    )

    medicines = prepare_medications(medication_rows)
    procedures = prepare_procedures(
        procedure_rows,
        encounter_base_cost,
    )
    observations = prepare_observations(observation_rows)

    consultation_charge = round(max(encounter_base_cost, 850), 2)

    procedure_total = round(
        sum(item["amount"] for item in procedures),
        2,
    )

    medicine_total = round(
        sum(item["total"] for item in medicines),
        2,
    )

    encounter_class = get_value(
        encounter,
        "ENCOUNTERCLASS",
        default="outpatient",
    ).lower()

    admission_days = 1

    start_date = parse_date(encounter_start)
    stop_date = parse_date(encounter_stop)

    if start_date is not None and stop_date is not None:
        admission_days = max((stop_date.date() - start_date.date()).days, 1)

    room_charge = (
        round(admission_days * 3500, 2)
        if encounter_class in {"inpatient", "emergency", "urgentcare"}
        else 0.0
    )

    nursing_charge = (
        round(admission_days * 900, 2)
        if room_charge > 0
        else 0.0
    )

    diagnostic_charge = round(
        max(len(observations), 1) * 350,
        2,
    )

    hospital_subtotal = round(
        consultation_charge
        + procedure_total
        + room_charge
        + nursing_charge
        + diagnostic_charge,
        2,
    )

    hospital_discount = round(hospital_subtotal * 0.03, 2)
    hospital_total = round(hospital_subtotal - hospital_discount, 2)

    supported_claim_amount = round(
        hospital_total + medicine_total,
        2,
    )

    synthea_claim_cost = safe_float(
        get_value(encounter, "TOTAL_CLAIM_COST"),
        default=0.0,
    )

    if synthea_claim_cost > 0:
        supported_claim_amount = round(
            max(supported_claim_amount, synthea_claim_cost),
            2,
        )

    claimed_amount = supported_claim_amount

    expected_action = "ELIGIBLE_FOR_AUTO_APPROVAL"
    expected_failures: list[str] = []

    if scenario == "UNPRESCRIBED_MEDICINE":
        extra_item = {
            "name": "Premium Multivitamin Supplement",
            "strength": "1 tablet",
            "dosage": "1 tablet",
            "frequency": "Once daily",
            "duration": "30 days",
            "instructions": "After meals",
            "quantity": 30,
            "unit_price": 35.0,
            "total": 1050.0,
            "prescribed": False,
        }

        medicines.append(extra_item)
        medicine_total = round(medicine_total + extra_item["total"], 2)
        supported_claim_amount = round(hospital_total + medicine_total, 2)
        claimed_amount = supported_claim_amount

        expected_action = "MEDICAL_REVIEW"
        expected_failures.append("PRESCRIPTION_PHARMACY_MISMATCH")

    elif scenario == "LOW_QUALITY_HANDWRITING":
        expected_action = "MANUAL_DOCUMENT_REVIEW"
        expected_failures.append("LOW_HANDWRITING_CONFIDENCE")

    elif scenario == "MISSING_DISCHARGE_SUMMARY":
        expected_action = "INCOMPLETE_CLAIM"
        expected_failures.append("MANDATORY_DOCUMENT_MISSING")

    elif scenario == "CLAIM_AMOUNT_MISMATCH":
        claimed_amount = round(supported_claim_amount + 2750, 2)
        expected_action = "FINANCIAL_REVIEW"
        expected_failures.append("CLAIM_AMOUNT_MISMATCH")

    policy_number = (
        f"POL-{patient_id.replace('-', '')[:8].upper()}"
        if patient_id
        else f"POL-{claim_number:06d}"
    )

    claim_id = f"CLM-{claim_number:05d}"

    hospital_name = get_value(
        organization,
        "NAME",
        default="Synthea General Hospital",
    )

    doctor_name = get_value(
        provider,
        "NAME",
        default="Dr. Alex Morgan",
    )

    payer_name = get_value(
        payer,
        "NAME",
        default="Synthetic Health Insurance",
    )

    claim_type = (
        "HOSPITALIZATION"
        if encounter_class in {"inpatient", "emergency", "urgentcare"}
        else "OUTPATIENT"
    )

    return {
        "claim_id": claim_id,
        "patient_id": patient_id,
        "encounter_id": encounter_id,
        "scenario": scenario,
        "expected_action": expected_action,
        "expected_failures": expected_failures,
        "patient": {
            "name": patient_name,
            "birthdate": format_date(get_value(patient, "BIRTHDATE")),
            "age": age,
            "gender": get_value(patient, "GENDER"),
            "address": get_value(patient, "ADDRESS"),
            "city": get_value(patient, "CITY"),
            "state": get_value(patient, "STATE"),
            "zip": get_value(patient, "ZIP"),
            "phone": get_value(patient, "PHONE"),
            "policy_number": policy_number,
            "payer_name": payer_name,
        },
        "encounter": {
            "type": encounter_class.title(),
            "claim_type": claim_type,
            "description": get_value(
                encounter,
                "DESCRIPTION",
                default=encounter_class.title(),
            ),
            "admission_date": format_datetime(encounter_start),
            "discharge_date": format_datetime(encounter_stop),
            "hospital_name": hospital_name,
            "doctor_name": doctor_name,
            "diagnosis": diagnosis,
            "admission_days": admission_days,
        },
        "medications": medicines,
        "procedures": procedures,
        "observations": observations,
        "financial": {
            "consultation_charge": consultation_charge,
            "room_charge": room_charge,
            "nursing_charge": nursing_charge,
            "diagnostic_charge": diagnostic_charge,
            "procedure_charge": procedure_total,
            "hospital_subtotal": hospital_subtotal,
            "hospital_discount": hospital_discount,
            "hospital_total": hospital_total,
            "pharmacy_total": medicine_total,
            "supported_claim_amount": supported_claim_amount,
            "claimed_amount": claimed_amount,
            "payer_coverage": safe_float(
                get_value(encounter, "PAYER_COVERAGE"),
                default=supported_claim_amount * 0.8,
            ),
        },
    }


# ============================================================
# DOCUMENT GENERATORS
# ============================================================

def generate_claim_form(
    claim: dict[str, Any],
    output_path: Path,
) -> None:
    patient = claim["patient"]
    encounter = claim["encounter"]
    financial = claim["financial"]

    story = [
        Paragraph("HEALTH INSURANCE CLAIM FORM", TITLE_STYLE),
        Paragraph(
            "Synthetic document generated from Synthea patient data",
            CENTER_STYLE,
        ),
        Spacer(1, 8),
        information_table(
            [
                ("Claim ID", claim["claim_id"]),
                ("Claim Type", encounter["claim_type"]),
                ("Policy Number", patient["policy_number"]),
                ("Insurance Provider", patient["payer_name"]),
                ("Patient ID", claim["patient_id"]),
                ("Patient Name", patient["name"]),
                ("Date of Birth", patient["birthdate"]),
                ("Age", patient["age"]),
                ("Gender", patient["gender"]),
                ("Hospital", encounter["hospital_name"]),
                ("Attending Doctor", encounter["doctor_name"]),
                ("Admission Date", encounter["admission_date"]),
                ("Discharge Date", encounter["discharge_date"]),
                ("Primary Diagnosis", encounter["diagnosis"]),
                ("Claimed Amount", f"INR {financial['claimed_amount']:,.2f}"),
            ]
        ),
        Spacer(1, 15),
        Paragraph("Patient Declaration", SUBTITLE_STYLE),
        Paragraph(
            "I confirm that the information and supporting documents "
            "submitted with this reimbursement claim are complete and correct.",
            BODY_STYLE,
        ),
        Spacer(1, 30),
        Table(
            [
                [
                    Paragraph("Patient Signature: ____________________", BODY_STYLE),
                    Paragraph("Date: ____________________", BODY_STYLE),
                ]
            ],
            colWidths=[100 * mm, 55 * mm],
        ),
    ]

    make_pdf(output_path, story, "Healthcare Claim Form")


def find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    possible_fonts = [
        "C:/Windows/Fonts/segoepr.ttf",
        "C:/Windows/Fonts/comic.ttf",
        "C:/Windows/Fonts/inkfree.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Italic.ttf",
    ]

    for font_path in possible_fonts:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size)
            except OSError:
                continue

    return ImageFont.load_default()


def handwritten_text(
    image: Image.Image,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    line_spacing: int = 12,
) -> None:
    draw = ImageDraw.Draw(image)

    words = text.split()
    lines: list[str] = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        bounding_box = draw.textbbox((0, 0), test_line, font=font)
        width = bounding_box[2] - bounding_box[0]

        if width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    x, y = position

    for line in lines:
        x_jitter = random.randint(-4, 4)
        y_jitter = random.randint(-2, 2)

        draw.text(
            (x + x_jitter, y + y_jitter),
            line,
            font=font,
            fill=(20, 45, 95),
        )

        bounding_box = draw.textbbox((0, 0), line, font=font)
        line_height = bounding_box[3] - bounding_box[1]
        y += line_height + line_spacing


def generate_handwritten_prescription(
    claim: dict[str, Any],
    output_path: Path,
) -> None:
    width = 1240
    height = 1754

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    header_font = find_font(42)
    label_font = find_font(25)
    handwriting_font = find_font(31)
    medicine_font = find_font(29)

    draw.rectangle(
        (40, 40, width - 40, height - 40),
        outline=(55, 90, 120),
        width=4,
    )

    draw.text(
        (80, 75),
        claim["encounter"]["hospital_name"],
        font=header_font,
        fill=(20, 65, 100),
    )

    draw.text(
        (80, 145),
        "MEDICAL PRESCRIPTION",
        font=header_font,
        fill=(25, 25, 25),
    )

    draw.line((80, 215, width - 80, 215), fill=(80, 80, 80), width=2)

    label_y = 255

    labels = [
        ("Patient:", claim["patient"]["name"]),
        ("Age / Gender:", f"{claim['patient']['age']} / {claim['patient']['gender']}"),
        ("Date:", claim["encounter"]["admission_date"].split(" ")[0]),
        ("Doctor:", claim["encounter"]["doctor_name"]),
        ("Diagnosis:", claim["encounter"]["diagnosis"]),
    ]

    for label, value in labels:
        draw.text(
            (90, label_y),
            label,
            font=label_font,
            fill=(60, 60, 60),
        )

        handwritten_text(
            image,
            (330, label_y - 3),
            str(value),
            handwriting_font,
            max_width=780,
        )

        label_y += 75

    draw.line((80, label_y, width - 80, label_y), fill=(150, 150, 150), width=2)
    label_y += 35

    draw.text(
        (90, label_y),
        "Rx",
        font=header_font,
        fill=(20, 50, 100),
    )

    label_y += 75

    prescribed_medications = [
        medication
        for medication in claim["medications"]
        if medication.get("prescribed", True)
    ]

    for index, medicine in enumerate(prescribed_medications, start=1):
        medicine_text = (
            f"{index}. {medicine['name']} {medicine['strength']}\n"
            f"   {medicine['dosage']}, {medicine['frequency']}, "
            f"{medicine['duration']} - {medicine['instructions']}"
        )

        handwritten_text(
            image,
            (125, label_y),
            medicine_text,
            medicine_font,
            max_width=1000,
            line_spacing=10,
        )

        label_y += 145

        if label_y > 1400:
            break

    handwritten_text(
        image,
        (100, 1460),
        "Follow-up as advised. Continue adequate hydration.",
        medicine_font,
        max_width=900,
    )

    draw.line(
        (790, 1600, 1110, 1600),
        fill=(60, 60, 60),
        width=2,
    )

    draw.text(
        (835, 1615),
        "Doctor Signature",
        font=label_font,
        fill=(60, 60, 60),
    )

    handwritten_text(
        image,
        (845, 1550),
        claim["encounter"]["doctor_name"],
        handwriting_font,
        max_width=280,
    )

    if claim["scenario"] == "LOW_QUALITY_HANDWRITING":
        image = image.filter(ImageFilter.GaussianBlur(radius=1.4))

        noise_overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
        noise_draw = ImageDraw.Draw(noise_overlay)

        for _ in range(1400):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            opacity = random.randint(10, 35)
            noise_draw.point((x, y), fill=(30, 30, 30, opacity))

        image = Image.alpha_composite(
            image.convert("RGBA"),
            noise_overlay,
        ).convert("RGB")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=90)


def generate_hospital_invoice(
    claim: dict[str, Any],
    output_path: Path,
) -> None:
    encounter = claim["encounter"]
    patient = claim["patient"]
    financial = claim["financial"]

    invoice_rows: list[list[Any]] = [
        [
            "Consultation charges",
            "1",
            f"{financial['consultation_charge']:,.2f}",
            f"{financial['consultation_charge']:,.2f}",
        ]
    ]

    if financial["room_charge"] > 0:
        invoice_rows.append(
            [
                "Room charges",
                str(encounter["admission_days"]),
                "3,500.00",
                f"{financial['room_charge']:,.2f}",
            ]
        )

    if financial["nursing_charge"] > 0:
        invoice_rows.append(
            [
                "Nursing charges",
                str(encounter["admission_days"]),
                "900.00",
                f"{financial['nursing_charge']:,.2f}",
            ]
        )

    for procedure in claim["procedures"]:
        invoice_rows.append(
            [
                procedure["description"],
                procedure["quantity"],
                f"{procedure['rate']:,.2f}",
                f"{procedure['amount']:,.2f}",
            ]
        )

    invoice_rows.append(
        [
            "Diagnostic and laboratory services",
            "1",
            f"{financial['diagnostic_charge']:,.2f}",
            f"{financial['diagnostic_charge']:,.2f}",
        ]
    )

    story = [
        Paragraph(encounter["hospital_name"], TITLE_STYLE),
        Paragraph("DETAILED HOSPITAL INVOICE", CENTER_STYLE),
        Spacer(1, 10),
        information_table(
            [
                ("Invoice Number", f"HINV-{claim['claim_id'][4:]}"),
                ("Claim ID", claim["claim_id"]),
                ("Patient Name", patient["name"]),
                ("Patient ID", claim["patient_id"]),
                ("Admission Date", encounter["admission_date"]),
                ("Discharge Date", encounter["discharge_date"]),
                ("Diagnosis", encounter["diagnosis"]),
            ]
        ),
        Spacer(1, 12),
        line_item_table(
            ["Description", "Quantity", "Rate (INR)", "Amount (INR)"],
            invoice_rows,
            [88 * mm, 22 * mm, 30 * mm, 32 * mm],
        ),
        Spacer(1, 12),
        information_table(
            [
                ("Subtotal", f"INR {financial['hospital_subtotal']:,.2f}"),
                ("Discount", f"INR {financial['hospital_discount']:,.2f}"),
                ("Hospital Invoice Total", f"INR {financial['hospital_total']:,.2f}"),
            ]
        ),
    ]

    make_pdf(output_path, story, "Hospital Invoice")


def generate_pharmacy_invoice(
    claim: dict[str, Any],
    output_path: Path,
) -> None:
    rows = []

    for medicine in claim["medications"]:
        rows.append(
            [
                medicine["name"],
                medicine["strength"],
                medicine["quantity"],
                f"{medicine['unit_price']:,.2f}",
                f"{medicine['total']:,.2f}",
            ]
        )

    story = [
        Paragraph("HEALTHPLUS PHARMACY", TITLE_STYLE),
        Paragraph("PHARMACY TAX INVOICE", CENTER_STYLE),
        Spacer(1, 10),
        information_table(
            [
                ("Invoice Number", f"PHM-{claim['claim_id'][4:]}"),
                ("Claim ID", claim["claim_id"]),
                ("Patient Name", claim["patient"]["name"]),
                ("Prescription Date", claim["encounter"]["admission_date"]),
                ("Prescribing Doctor", claim["encounter"]["doctor_name"]),
            ]
        ),
        Spacer(1, 12),
        line_item_table(
            ["Medicine", "Strength", "Qty", "Unit Price", "Amount"],
            rows,
            [69 * mm, 30 * mm, 16 * mm, 28 * mm, 30 * mm],
        ),
        Spacer(1, 12),
        information_table(
            [
                (
                    "Pharmacy Invoice Total",
                    f"INR {claim['financial']['pharmacy_total']:,.2f}",
                ),
            ]
        ),
    ]

    make_pdf(output_path, story, "Pharmacy Invoice")


def generate_discharge_summary(
    claim: dict[str, Any],
    output_path: Path,
) -> None:
    medication_list = "<br/>".join(
        (
            f"{index}. {medicine['name']} {medicine['strength']} - "
            f"{medicine['frequency']} for {medicine['duration']}"
        )
        for index, medicine in enumerate(
            [
                medicine
                for medicine in claim["medications"]
                if medicine.get("prescribed", True)
            ],
            start=1,
        )
    )

    procedure_list = "<br/>".join(
        f"{index}. {procedure['description']}"
        for index, procedure in enumerate(
            claim["procedures"],
            start=1,
        )
    )

    story = [
        Paragraph(claim["encounter"]["hospital_name"], TITLE_STYLE),
        Paragraph("DISCHARGE SUMMARY", CENTER_STYLE),
        Spacer(1, 10),
        information_table(
            [
                ("Claim ID", claim["claim_id"]),
                ("Patient Name", claim["patient"]["name"]),
                ("Patient ID", claim["patient_id"]),
                ("Admission Date", claim["encounter"]["admission_date"]),
                ("Discharge Date", claim["encounter"]["discharge_date"]),
                ("Attending Doctor", claim["encounter"]["doctor_name"]),
                ("Final Diagnosis", claim["encounter"]["diagnosis"]),
            ]
        ),
        Spacer(1, 12),
        Paragraph("Clinical Summary", SUBTITLE_STYLE),
        Paragraph(
            (
                f"The patient was evaluated and treated for "
                f"<b>{claim['encounter']['diagnosis']}</b>. "
                f"The clinical condition improved following treatment "
                f"and the patient was discharged in stable condition."
            ),
            BODY_STYLE,
        ),
        Paragraph("Procedures and Treatment", SUBTITLE_STYLE),
        Paragraph(procedure_list or "No procedures recorded.", BODY_STYLE),
        Paragraph("Discharge Medications", SUBTITLE_STYLE),
        Paragraph(medication_list, BODY_STYLE),
        Paragraph("Follow-up Advice", SUBTITLE_STYLE),
        Paragraph(
            "Take medicines as prescribed, maintain adequate hydration, "
            "and follow up with the treating physician if symptoms persist.",
            BODY_STYLE,
        ),
    ]

    make_pdf(output_path, story, "Discharge Summary")


def generate_diagnostic_report(
    claim: dict[str, Any],
    output_path: Path,
) -> None:
    rows = []

    for observation in claim["observations"]:
        result = observation["value"]

        if observation["units"]:
            result = f"{result} {observation['units']}"

        rows.append(
            [
                observation["date"],
                observation["test_name"],
                result,
                "Review with treating physician",
            ]
        )

    story = [
        Paragraph("SYNTHEA DIAGNOSTIC LABORATORY", TITLE_STYLE),
        Paragraph("DIAGNOSTIC REPORT", CENTER_STYLE),
        Spacer(1, 10),
        information_table(
            [
                ("Claim ID", claim["claim_id"]),
                ("Patient Name", claim["patient"]["name"]),
                ("Patient ID", claim["patient_id"]),
                ("Ordering Doctor", claim["encounter"]["doctor_name"]),
                ("Hospital", claim["encounter"]["hospital_name"]),
                ("Diagnosis", claim["encounter"]["diagnosis"]),
            ]
        ),
        Spacer(1, 12),
        line_item_table(
            ["Date", "Test / Observation", "Result", "Reference / Comment"],
            rows,
            [27 * mm, 64 * mm, 35 * mm, 47 * mm],
        ),
        Spacer(1, 12),
        Paragraph(
            "This is a synthetic diagnostic report generated for demonstration.",
            SMALL_STYLE,
        ),
    ]

    make_pdf(output_path, story, "Diagnostic Report")


def generate_payment_receipt(
    claim: dict[str, Any],
    output_path: Path,
) -> None:
    amount_paid = claim["financial"]["supported_claim_amount"]

    story = [
        Paragraph(claim["encounter"]["hospital_name"], TITLE_STYLE),
        Paragraph("PAYMENT RECEIPT", CENTER_STYLE),
        Spacer(1, 14),
        information_table(
            [
                ("Receipt Number", f"RCPT-{claim['claim_id'][4:]}"),
                ("Claim ID", claim["claim_id"]),
                ("Patient Name", claim["patient"]["name"]),
                ("Hospital Invoice", f"HINV-{claim['claim_id'][4:]}"),
                ("Pharmacy Invoice", f"PHM-{claim['claim_id'][4:]}"),
                ("Amount Paid", f"INR {amount_paid:,.2f}"),
                ("Payment Method", random.choice(["Card", "UPI", "Bank Transfer"])),
                (
                    "Transaction Reference",
                    f"TXN{random.randint(100000000, 999999999)}",
                ),
                ("Payment Status", "PAID"),
            ]
        ),
        Spacer(1, 18),
        Paragraph(
            "This receipt confirms payment against the submitted medical invoices.",
            BODY_STYLE,
        ),
    ]

    make_pdf(output_path, story, "Payment Receipt")


# ============================================================
# CLAIM PACKAGE GENERATION
# ============================================================

def generate_claim_package(
    claim: dict[str, Any],
    patient_folder: Path,
) -> list[str]:
    patient_folder.mkdir(parents=True, exist_ok=True)

    generated_documents: list[str] = []

    document_generators = [
        (
            "claim_form.pdf",
            generate_claim_form,
        ),
        (
            "handwritten_prescription.png",
            generate_handwritten_prescription,
        ),
        (
            "hospital_invoice.pdf",
            generate_hospital_invoice,
        ),
        (
            "pharmacy_invoice.pdf",
            generate_pharmacy_invoice,
        ),
        (
            "diagnostic_report.pdf",
            generate_diagnostic_report,
        ),
        (
            "payment_receipt.pdf",
            generate_payment_receipt,
        ),
    ]

    if claim["scenario"] != "MISSING_DISCHARGE_SUMMARY":
        document_generators.append(
            (
                "discharge_summary.pdf",
                generate_discharge_summary,
            )
        )

    for filename, generator in document_generators:
        output_path = patient_folder / filename
        generator(claim, output_path)
        generated_documents.append(filename)

    manifest = {
        "claim_id": claim["claim_id"],
        "patient_id": claim["patient_id"],
        "patient_name": claim["patient"]["name"],
        "encounter_id": claim["encounter_id"],
        "claim_type": claim["encounter"]["claim_type"],
        "scenario": claim["scenario"],
        "expected_action": claim["expected_action"],
        "expected_failures": claim["expected_failures"],
        "claimed_amount": claim["financial"]["claimed_amount"],
        "supported_claim_amount": claim["financial"]["supported_claim_amount"],
        "documents": generated_documents,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(
        patient_folder / "manifest.json",
        "w",
        encoding="utf-8",
    ) as manifest_file:
        json.dump(
            manifest,
            manifest_file,
            indent=2,
            ensure_ascii=False,
        )

    return generated_documents


def main() -> None:
    print("=" * 70)
    print("Synthetic Healthcare Claims Document Generator")
    print("=" * 70)
    print(f"Project directory : {BASE_DIR}")
    print(f"Source CSV folder : {INPUT_DIR}")
    print(f"Output folder     : {OUTPUT_DIR}")
    print()

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            "The Synthea source folder was not found.\n"
            f"Expected location: {INPUT_DIR}\n\n"
            "Your project structure should be:\n"
            "sample_healthcare_dataset/\n"
            "├── generate_claims_document.py\n"
            "└── synthea_source_dataset/\n"
            "    ├── patients.csv\n"
            "    ├── encounters.csv\n"
            "    └── ..."
        )

    required_files = [
        "patients.csv",
        "encounters.csv",
    ]

    missing_files = [
        filename
        for filename in required_files
        if not (INPUT_DIR / filename).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "The following required CSV files are missing from "
            f"{INPUT_DIR}:\n"
            + "\n".join(f"- {filename}" for filename in missing_files)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_synthea_data()

    patients = data["patients"].copy()
    encounters = data["encounters"].copy()

    if patients.empty:
        raise ValueError("patients.csv does not contain any records.")

    if encounters.empty:
        raise ValueError("encounters.csv does not contain any records.")

    if MAX_PATIENTS is not None:
        patients = patients.head(MAX_PATIENTS)

    claims_index: list[dict[str, Any]] = []
    generated_count = 0

    for _, patient in patients.iterrows():
        patient_id = get_value(patient, "ID")

        if not patient_id:
            print("Skipping a patient record because ID is missing.")
            continue

        encounter = choose_encounter(
            patient_id=patient_id,
            encounters=encounters,
        )

        if encounter is None:
            print(
                f"Skipping patient {patient_id}: "
                "no associated encounter was found."
            )
            continue

        if GENERATE_TEST_SCENARIOS:
            scenario = TEST_SCENARIOS[
                generated_count % len(TEST_SCENARIOS)
            ]
        else:
            scenario = "VALID_CLAIM"

        claim_number = 1001 + generated_count

        claim = build_claim_record(
            claim_number=claim_number,
            patient=patient,
            encounter=encounter,
            data=data,
            scenario=scenario,
        )

        patient_name = claim["patient"]["name"]
        short_patient_id = claim["patient_id"][:8]

        patient_folder_name = sanitize_filename(
            f"{patient_name}_{short_patient_id}"
        )

        patient_folder = OUTPUT_DIR / patient_folder_name

        generated_documents = generate_claim_package(
            claim=claim,
            patient_folder=patient_folder,
        )

        claims_index.append(
            {
                "CLAIM_ID": claim["claim_id"],
                "PATIENT_ID": claim["patient_id"],
                "PATIENT_NAME": patient_name,
                "ENCOUNTER_ID": claim["encounter_id"],
                "CLAIM_TYPE": claim["encounter"]["claim_type"],
                "SCENARIO": claim["scenario"],
                "EXPECTED_ACTION": claim["expected_action"],
                "EXPECTED_FAILURES": "|".join(
                    claim["expected_failures"]
                ),
                "CLAIMED_AMOUNT": claim["financial"]["claimed_amount"],
                "SUPPORTED_AMOUNT": (
                    claim["financial"]["supported_claim_amount"]
                ),
                "DOCUMENT_COUNT": len(generated_documents),
                "PATIENT_FOLDER": str(patient_folder),
            }
        )

        generated_count += 1

        print(
            f"Generated {claim['claim_id']} | "
            f"{patient_name} | "
            f"{scenario} | "
            f"{len(generated_documents)} documents"
        )

    if not claims_index:
        raise RuntimeError(
            "No claim packages were generated. "
            "Check whether patient IDs in patients.csv match "
            "the PATIENT column in encounters.csv."
        )

    claims_index_df = pd.DataFrame(claims_index)

    claims_index_path = OUTPUT_DIR / "claims_index.csv"

    claims_index_df.to_csv(
        claims_index_path,
        index=False,
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("Generation completed successfully")
    print("=" * 70)
    print(f"Claims generated : {generated_count}")
    print(f"Output folder    : {OUTPUT_DIR}")
    print(f"Claims index     : {claims_index_path}")

if __name__ == "__main__":
    main()