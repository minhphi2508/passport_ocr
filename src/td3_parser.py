from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any


TD3_LINE_LENGTH = 44
FILLER = "<"


@dataclass
class TD3Fields:
    document_type: str
    issuing_country: str
    surname: str
    given_names: str

    passport_number: str
    passport_number_check_digit: str

    nationality: str
    birth_date_raw: str
    birth_date: str | None
    birth_date_check_digit: str

    sex: str

    expiry_date_raw: str
    expiry_date: str | None
    expiry_date_check_digit: str

    personal_number: str
    personal_number_check_digit: str

    final_check_digit: str

    line_1: str
    line_2: str

    parse_status: str
    parse_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_mrz_line(line: str) -> str:
    """
    Làm sạch một dòng MRZ nhưng không tự sửa lỗi OCR.
    """
    cleaned = str(line).upper().strip()

    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace("«", FILLER)
    cleaned = cleaned.replace("‹", FILLER)
    cleaned = cleaned.replace("＜", FILLER)

    return cleaned


def remove_fillers(value: str) -> str:
    """
    Xóa filler ở đầu/cuối và đổi filler nội bộ thành khoảng trắng.
    """
    return " ".join(
        part
        for part in value.strip(FILLER).split(FILLER)
        if part
    )

def correct_alpha_field_ocr(
    value: str,
) -> str:
    """
    Sửa OCR trong các field ICAO chỉ cho phép chữ cái.

    Chỉ dùng cho:
    - surname / given names
    - nationality
    - issuing country

    Không dùng cho passport number.
    """

    replacements = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "6": "G",
        "8": "B",
    }

    return "".join(
        replacements.get(char, char)
        for char in value
    )


def parse_name_field(
    name_field: str,
) -> tuple[str, str]:
    """
    TD3 line 1 lưu tên theo dạng:

    SURNAME<<GIVEN<NAMES<<<<
    """
    if "<<" in name_field:
        surname_raw, given_names_raw = name_field.split(
            "<<",
            maxsplit=1,
        )
    else:
        surname_raw = name_field
        given_names_raw = ""

    surname = remove_fillers(
        correct_alpha_field_ocr(
            surname_raw
        )
    )

    given_names = remove_fillers(
        correct_alpha_field_ocr(
            given_names_raw
        )
    )

    return surname, given_names


def parse_td3_date(
    value: str,
    field_type: str,
    reference_date: date | None = None,
) -> str | None:
    """
    Chuyển YYMMDD thành YYYY-MM-DD.

    Với ngày sinh:
    - ưu tiên một ngày không nằm trong tương lai;
    - thường chọn thế kỷ gần nhất hợp lý.

    Với ngày hết hạn:
    - ưu tiên năm trong khoảng tương đối gần hiện tại.

    Đây chỉ là cách suy luận thế kỷ, vì MRZ chỉ chứa hai chữ số năm.
    """
    if reference_date is None:
        reference_date = date.today()

    if len(value) != 6 or not value.isdigit():
        return None

    year_2 = int(value[0:2])
    month = int(value[2:4])
    day = int(value[4:6])

    candidate_years = [
        1900 + year_2,
        2000 + year_2,
        2100 + year_2,
    ]

    valid_candidates: list[date] = []

    for candidate_year in candidate_years:
        try:
            candidate_date = date(
                candidate_year,
                month,
                day,
            )
        except ValueError:
            continue

        valid_candidates.append(candidate_date)

    if not valid_candidates:
        return None

    if field_type == "birth":
        non_future = [
            candidate
            for candidate in valid_candidates
            if candidate <= reference_date
        ]

        if not non_future:
            return None

        # Chọn ngày gần hiện tại nhất nhưng vẫn không nằm trong tương lai.
        selected = max(non_future)

    elif field_type == "expiry":
        # Hộ chiếu có thể đã hết hạn hoặc còn hạn.
        # Chọn ngày gần reference_date nhất.
        selected = min(
            valid_candidates,
            key=lambda candidate: abs(
                (candidate - reference_date).days
            ),
        )

    else:
        raise ValueError(
            f"field_type không hợp lệ: {field_type}"
        )

    return selected.isoformat()


def validate_line_lengths(
    line_1: str,
    line_2: str,
) -> None:
    if len(line_1) != TD3_LINE_LENGTH:
        raise ValueError(
            f"Line 1 phải dài 44 ký tự, hiện tại là {len(line_1)}."
        )

    if len(line_2) != TD3_LINE_LENGTH:
        raise ValueError(
            f"Line 2 phải dài 44 ký tự, hiện tại là {len(line_2)}."
        )


def parse_td3(
    line_1: str,
    line_2: str,
    strict_length: bool = True,
    reference_date: date | None = None,
) -> TD3Fields:
    """
    Parse MRZ TD3 gồm hai dòng, mỗi dòng 44 ký tự.

    strict_length=True:
        báo lỗi nếu một trong hai dòng không đúng 44 ký tự.

    strict_length=False:
        vẫn cố parse bằng cách pad hoặc cắt về 44 ký tự.
        Chế độ này hữu ích với output OCR chưa hoàn hảo.
    """
    cleaned_line_1 = clean_mrz_line(line_1)
    cleaned_line_2 = clean_mrz_line(line_2)

    parse_status = "success"
    parse_error: str | None = None

    try:
        if strict_length:
            validate_line_lengths(
                cleaned_line_1,
                cleaned_line_2,
            )
        else:
            cleaned_line_1 = (
                cleaned_line_1[:TD3_LINE_LENGTH]
                .ljust(TD3_LINE_LENGTH, FILLER)
            )

            cleaned_line_2 = (
                cleaned_line_2[:TD3_LINE_LENGTH]
                .ljust(TD3_LINE_LENGTH, FILLER)
            )

        document_type = remove_fillers(
            cleaned_line_1[0:2]
        )

        issuing_country = remove_fillers(
            correct_alpha_field_ocr(
                cleaned_line_1[2:5]
             )
        )       

        name_field = cleaned_line_1[5:44]

        surname, given_names = parse_name_field(
            name_field
        )

        passport_number = remove_fillers(
            cleaned_line_2[0:9]
        )

        passport_number_check_digit = (
            cleaned_line_2[9]
        )

        nationality = remove_fillers(
            correct_alpha_field_ocr(
                cleaned_line_2[10:13]
            )
        )

        birth_date_raw = cleaned_line_2[13:19]

        birth_date_check_digit = (
            cleaned_line_2[19]
        )

        sex = remove_fillers(
            cleaned_line_2[20]
        )

        expiry_date_raw = cleaned_line_2[21:27]

        expiry_date_check_digit = (
            cleaned_line_2[27]
        )

        personal_number = remove_fillers(
            cleaned_line_2[28:42]
        )

        personal_number_check_digit = (
            cleaned_line_2[42]
        )

        final_check_digit = cleaned_line_2[43]

        birth_date = parse_td3_date(
            birth_date_raw,
            field_type="birth",
            reference_date=reference_date,
        )

        expiry_date = parse_td3_date(
            expiry_date_raw,
            field_type="expiry",
            reference_date=reference_date,
        )

    except Exception as error:
        parse_status = "error"
        parse_error = repr(error)

        document_type = ""
        issuing_country = ""
        surname = ""
        given_names = ""

        passport_number = ""
        passport_number_check_digit = ""

        nationality = ""
        birth_date_raw = ""
        birth_date = None
        birth_date_check_digit = ""

        sex = ""

        expiry_date_raw = ""
        expiry_date = None
        expiry_date_check_digit = ""

        personal_number = ""
        personal_number_check_digit = ""

        final_check_digit = ""

    return TD3Fields(
        document_type=document_type,
        issuing_country=issuing_country,
        surname=surname,
        given_names=given_names,
        passport_number=passport_number,
        passport_number_check_digit=passport_number_check_digit,
        nationality=nationality,
        birth_date_raw=birth_date_raw,
        birth_date=birth_date,
        birth_date_check_digit=birth_date_check_digit,
        sex=sex,
        expiry_date_raw=expiry_date_raw,
        expiry_date=expiry_date,
        expiry_date_check_digit=expiry_date_check_digit,
        personal_number=personal_number,
        personal_number_check_digit=personal_number_check_digit,
        final_check_digit=final_check_digit,
        line_1=cleaned_line_1,
        line_2=cleaned_line_2,
        parse_status=parse_status,
        parse_error=parse_error,
    )


def main() -> None:
    """
    Ví dụ chuẩn TD3 để kiểm tra parser.
    """
    example_line_1 = (
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
    )

    example_line_2 = (
        "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
    )

    result = parse_td3(
        example_line_1,
        example_line_2,
        strict_length=True,
        reference_date=date(2026, 8, 1),
    )

    print("Kết quả parse TD3:")
    print()

    for key, value in result.to_dict().items():
        print(f"{key:<32}: {value}")


if __name__ == "__main__":
    main()