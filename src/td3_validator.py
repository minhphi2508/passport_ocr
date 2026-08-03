from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


WEIGHTS = (7, 3, 1)


@dataclass
class TD3ValidationResult:
    passport_number_check_valid: bool | None
    birth_date_check_valid: bool | None
    expiry_date_check_valid: bool | None
    personal_number_check_valid: bool | None
    final_check_valid: bool | None

    all_main_checks_valid: bool | None

    passport_number_expected_digit: str | None
    birth_date_expected_digit: str | None
    expiry_date_expected_digit: str | None
    personal_number_expected_digit: str | None
    final_expected_digit: str | None

    validation_status: str
    validation_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def character_value(character: str) -> int:
    """
    ICAO 9303:
    - 0–9 -> 0–9
    - A–Z -> 10–35
    - <   -> 0
    """
    if character.isdigit():
        return int(character)

    if "A" <= character <= "Z":
        return ord(character) - ord("A") + 10

    if character == "<":
        return 0

    raise ValueError(
        f"Ký tự không hợp lệ trong MRZ: {character!r}"
    )


def calculate_check_digit(value: str) -> str:
    total = 0

    for index, character in enumerate(value):
        weight = WEIGHTS[index % len(WEIGHTS)]
        total += character_value(character) * weight

    return str(total % 10)


def compare_check_digit(
    value: str,
    actual_digit: str,
) -> tuple[bool | None, str | None]:
    """
    Trả về:
    - valid
    - expected digit

    Nếu actual digit không phải 0–9 thì không thể kiểm tra.
    """
    if not value:
        return None, None

    if len(actual_digit) != 1 or not actual_digit.isdigit():
        return None, None

    expected_digit = calculate_check_digit(value)

    return expected_digit == actual_digit, expected_digit


def validate_td3_lines(
    line_1: str,
    line_2: str,
) -> TD3ValidationResult:
    """
    Validate trực tiếp hai dòng TD3.

    Lưu ý:
    - Không loại record.
    - Không sửa OCR.
    - Chỉ ghi nhận kết quả.
    """
    try:
        line_1 = str(line_1).upper().strip().replace(" ", "")
        line_2 = str(line_2).upper().strip().replace(" ", "")

        if len(line_1) != 44 or len(line_2) != 44:
            return TD3ValidationResult(
                passport_number_check_valid=None,
                birth_date_check_valid=None,
                expiry_date_check_valid=None,
                personal_number_check_valid=None,
                final_check_valid=None,
                all_main_checks_valid=None,
                passport_number_expected_digit=None,
                birth_date_expected_digit=None,
                expiry_date_expected_digit=None,
                personal_number_expected_digit=None,
                final_expected_digit=None,
                validation_status="skipped_invalid_length",
                validation_error=(
                    f"TD3 cần 44+44 ký tự, hiện tại là "
                    f"{len(line_1)}+{len(line_2)}."
                ),
            )

        passport_number_value = line_2[0:9]
        passport_number_digit = line_2[9]

        birth_date_value = line_2[13:19]
        birth_date_digit = line_2[19]

        expiry_date_value = line_2[21:27]
        expiry_date_digit = line_2[27]

        personal_number_value = line_2[28:42]
        personal_number_digit = line_2[42]

        final_value = (
            line_2[0:10]
            + line_2[13:20]
            + line_2[21:43]
        )
        final_digit = line_2[43]

        (
            passport_number_valid,
            passport_number_expected,
        ) = compare_check_digit(
            passport_number_value,
            passport_number_digit,
        )

        (
            birth_date_valid,
            birth_date_expected,
        ) = compare_check_digit(
            birth_date_value,
            birth_date_digit,
        )

        (
            expiry_date_valid,
            expiry_date_expected,
        ) = compare_check_digit(
            expiry_date_value,
            expiry_date_digit,
        )

        (
            personal_number_valid,
            personal_number_expected,
        ) = compare_check_digit(
            personal_number_value,
            personal_number_digit,
        )

        (
            final_valid,
            final_expected,
        ) = compare_check_digit(
            final_value,
            final_digit,
        )

        main_checks = [
            passport_number_valid,
            birth_date_valid,
            expiry_date_valid,
            final_valid,
        ]

        if any(value is None for value in main_checks):
            all_main_checks_valid = None
        else:
            all_main_checks_valid = all(main_checks)

        return TD3ValidationResult(
            passport_number_check_valid=passport_number_valid,
            birth_date_check_valid=birth_date_valid,
            expiry_date_check_valid=expiry_date_valid,
            personal_number_check_valid=personal_number_valid,
            final_check_valid=final_valid,
            all_main_checks_valid=all_main_checks_valid,
            passport_number_expected_digit=passport_number_expected,
            birth_date_expected_digit=birth_date_expected,
            expiry_date_expected_digit=expiry_date_expected,
            personal_number_expected_digit=personal_number_expected,
            final_expected_digit=final_expected,
            validation_status="success",
            validation_error=None,
        )

    except Exception as error:
        return TD3ValidationResult(
            passport_number_check_valid=None,
            birth_date_check_valid=None,
            expiry_date_check_valid=None,
            personal_number_check_valid=None,
            final_check_valid=None,
            all_main_checks_valid=None,
            passport_number_expected_digit=None,
            birth_date_expected_digit=None,
            expiry_date_expected_digit=None,
            personal_number_expected_digit=None,
            final_expected_digit=None,
            validation_status="error",
            validation_error=repr(error),
        )


def main() -> None:
    line_1 = (
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
    )

    line_2 = (
        "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
    )

    result = validate_td3_lines(
        line_1=line_1,
        line_2=line_2,
    )

    print("Kết quả validation:")
    print()

    for key, value in result.to_dict().items():
        print(f"{key:<40}: {value}")


if __name__ == "__main__":
    main()