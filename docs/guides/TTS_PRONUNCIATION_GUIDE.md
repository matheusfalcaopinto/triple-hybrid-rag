# TTS Pronunciation Best Practices Guide

> A comprehensive guide for voice agents to properly pronounce structured data including dates, times, numbers, currencies, and more.

---

## Table of Contents

1. [General Principles](#general-principles)
2. [Dates](#dates)
3. [Times](#times)
4. [Numbers](#numbers)
5. [Currency & Prices](#currency--prices)
6. [Phone Numbers](#phone-numbers)
7. [Email Addresses](#email-addresses)
8. [URLs & Web Addresses](#urls--web-addresses)
9. [Physical Addresses](#physical-addresses)
10. [Fractions & Decimals](#fractions--decimals)
11. [Percentages](#percentages)
12. [Ordinal Numbers](#ordinal-numbers)
13. [Abbreviations & Acronyms](#abbreviations--acronyms)
14. [Units of Measurement](#units-of-measurement)
15. [Special Characters](#special-characters)
16. [Credit Card Numbers](#credit-card-numbers)
17. [SSML Reference](#ssml-reference)
18. [Implementation Recommendations](#implementation-recommendations)

---

## General Principles

Before diving into specific data types, these foundational rules apply across all TTS scenarios:

| Principle | Description |
|-----------|-------------|
| **Avoid Ambiguity** | Write out ambiguous terms fully; TTS can interpret the same string multiple ways |
| **Punctuation Matters** | Commas = brief pauses, periods = longer breaks, question marks = rising intonation |
| **Chunking** | Break long text into shorter sentences for better intonation |
| **Language Matching** | Ensure voice language matches script language |
| **Use SSML** | When available, SSML provides fine-grained control over pronunciation |

---

## Dates

### The Problem

`10/12/2025` could be interpreted as October 12th (US) or December 10th (international).

### Best Practices

| Format | Recommendation | Example |
|--------|---------------|---------|
| **Full written** | ✅ Preferred | "October 12, 2025" or "12 October 2025" |
| **Numeric slash** | ❌ Avoid | "10/12/2025" |
| **Day ordinals** | Use spoken form | "November 9" → "November ninth" |

### Year Pronunciation Rules

| Year Range | Pronunciation Style | Example |
|------------|---------------------|---------|
| Before 2000 | Two separate numbers | 1999 → "nineteen ninety-nine" |
| 2000-2009 | Full number with "oh" | 2008 → "two thousand oh eight" |
| 2010+ | Either style works | 2025 → "twenty twenty-five" or "two thousand twenty-five" |

### SSML Example

```xml
<say-as interpret-as="date" format="mdy">10/12/2025</say-as>
<!-- Reads as: "October twelfth, twenty twenty-five" -->
```

### Text Normalization

```python
# Input: "10/12/2025"
# Output: "October twelfth, twenty twenty-five"

# Input: "2025-12-22"
# Output: "December twenty-second, twenty twenty-five"
```

---

## Times

### Best Practices

| Format | How TTS Reads It | Notes |
|--------|------------------|-------|
| `8:15` | "eight fifteen" | Standard format ✅ |
| `8:15 PM` | "eight fifteen p.m." | Standard format ✅ |
| `20:15` | "twenty fifteen" | Military/24-hour time |
| `8:00` | "eight o'clock" | On-the-hour |

### Recommendations

- For maximum control, write times as you want them spoken
- Use "in the morning" / "in the afternoon" instead of AM/PM for clarity

### SSML Example

```xml
<say-as interpret-as="time" format="hms12">2:30pm</say-as>
<!-- Reads as: "two thirty p.m." -->
```

### Text Normalization

```python
# Input: "14:30"
# Output: "two thirty p.m." or "fourteen thirty"

# Input: "9:00 AM"
# Output: "nine a.m." or "nine o'clock in the morning"
```

---

## Numbers

### Cardinal Numbers

| Scenario | Input | Recommended Output |
|----------|-------|-------------------|
| Standard | `100` | "one hundred" |
| Large | `1,234,567` | "one million, two hundred thirty-four thousand, five hundred sixty-seven" |
| Ambiguous | `100` | Write as "one hundred" for clarity |

### Reading Digit-by-Digit

Use spaces or commas between digits to force individual pronunciation:

| Input | How to Format | Output |
|-------|--------------|--------|
| PIN code | `1 2 3 4` | "one, two, three, four" |
| Slower reading | `1, 2, 3, 4` | "one... two... three... four" |

### Large IDs and Reference Numbers

For order IDs or long sequences, split the text:

```
# Instead of:
"Your order id is 123456789012345"

# Use:
"Your order id is 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5"
```

### SSML Examples

```xml
<!-- Read as cardinal -->
<say-as interpret-as="cardinal">123</say-as>
<!-- Output: "one hundred twenty-three" -->

<!-- Read digit by digit -->
<say-as interpret-as="digits">123</say-as>
<!-- Output: "one two three" -->
```

---

## Currency & Prices

### Best Practices

| Input | ❌ Avoid | ✅ Recommended |
|-------|---------|---------------|
| `$100` | Let TTS guess | "one hundred dollars" |
| `$10.50` | "$10.50" | "ten dollars and fifty cents" |
| `€99.99` | "€99.99" | "ninety-nine euros and ninety-nine cents" |
| `R$1.234,56` | Raw format | "mil duzentos e trinta e quatro reais e cinquenta e seis centavos" |

### Currency Symbol Handling

Some TTS APIs require explicit escape: `\$123.60` → "one hundred twenty-three dollars and sixty cents"

### SSML Example

```xml
<say-as interpret-as="currency">$99.99 USD</say-as>
<!-- Output: "ninety-nine US dollars and ninety-nine cents" -->
```

### Text Normalization

```python
def normalize_currency(value: str, locale: str = "en-US") -> str:
    """
    Examples:
    - "$100" → "one hundred dollars"
    - "R$1.234,56" → "mil duzentos e trinta e quatro reais e cinquenta e seis centavos"
    - "€50" → "fifty euros"
    """
    pass
```

---

## Phone Numbers

### The Challenge

- `9876543210` may be read as "nine billion..."
- Grouping and pacing are essential for clarity

### Best Practices

| Technique | Example |
|-----------|---------|
| **Add pauses** | `123-456-7890` → use commas for pauses |
| **Write as words** | "one two three, four five six, seven eight nine zero" |
| **Use SSML** | `interpret-as="telephone"` |

### Regional Formatting

| Region | Input | Output |
|--------|-------|--------|
| US | `(555) 123-4567` | "five five five, one two three, four five six seven" |
| Brazil | `+55 11 99999-1234` | "plus fifty-five, eleven, nine nine nine nine nine, one two three four" |
| UK | `+44 20 7946 0958` | "plus forty-four, twenty, seven nine four six, zero nine five eight" |

### For Repeated Digits

| Input | Natural Output |
|-------|---------------|
| `9988877766` | "double nine, triple eight, double seven, double six" |

### SSML Example

```xml
<say-as interpret-as="telephone">+1-555-123-4567</say-as>
```

---

## Email Addresses

### The Problem

`support@company.com` may be mispronounced without explicit formatting.

### Best Practices

**Always spell out email addresses phonetically:**

| Symbol | Say As |
|--------|--------|
| `@` | "at" |
| `.` | "dot" |
| `-` | "dash" or "hyphen" |
| `_` | "underscore" |

### Examples

| Input | Output |
|-------|--------|
| `john.doe@gmail.com` | "john dot doe at gmail dot com" |
| `support_team@company.co.uk` | "support underscore team at company dot co dot u k" |
| `info-help@example.org` | "info dash help at example dot org" |

### Text Normalization

```python
def normalize_email(email: str) -> str:
    return (email
        .replace("@", " at ")
        .replace(".", " dot ")
        .replace("-", " dash ")
        .replace("_", " underscore "))
```

---

## URLs & Web Addresses

### Best Practices

| Symbol | Say As |
|--------|--------|
| `/` | "slash" |
| `.` | "dot" |
| `://` | "colon slash slash" (or omit for "https") |
| `-` | "dash" |
| `_` | "underscore" |
| `?` | "question mark" (or describe as "with parameter") |
| `&` | "and" |

### Examples

| Input | Output |
|-------|--------|
| `example.com` | "example dot com" |
| `https://company.com/contact` | "company dot com slash contact" |
| `api.service.io/v2/users` | "api dot service dot io slash v two slash users" |

### SSML Consideration

When embedding URLs in SSML, escape `&` as `&amp;`:

```xml
<speak>Visit example.com?a=1&amp;b=2</speak>
```

---

## Physical Addresses

### Challenges

- Abbreviations: "St." could be "Street" or "Saint"
- Numbers in addresses
- Apartment/unit designations

### Best Practices

| Abbreviation | Expand To |
|--------------|-----------|
| St. | Street (or Saint based on context) |
| Ave. | Avenue |
| Blvd. | Boulevard |
| Apt. | Apartment |
| # | Number |

### Example Normalization

| Input | Output |
|-------|--------|
| `123 Main St., Apt. 4B` | "one twenty-three Main Street, Apartment four B" |
| `456 Oak Ave., Suite 100` | "four fifty-six Oak Avenue, Suite one hundred" |
| `789 5th Blvd.` | "seven eighty-nine Fifth Boulevard" |

---

## Fractions & Decimals

### Fractions

| Fraction | Pronunciation |
|----------|--------------|
| `1/2` | "one half" |
| `1/4` | "one quarter" or "one fourth" |
| `3/4` | "three quarters" |
| `2/9` | "two ninths" |
| `1 3/4` | "one and three quarters" |
| `334/11` | "three hundred thirty-four over eleven" (for complex fractions) |

### Decimals

| Decimal | Pronunciation |
|---------|--------------|
| `3.14` | "three point one four" |
| `0.5` | "zero point five" or "point five" |
| `66.37` | "sixty-six point three seven" (NOT "sixty-six point thirty-seven") |

> **Important**: After the decimal point, read each digit individually!

### SSML Example

```xml
<say-as interpret-as="fraction">3/4</say-as>
<!-- Output: "three quarters" -->
```

---

## Percentages

### Best Practices

| Input | Output |
|-------|--------|
| `50%` | "fifty percent" |
| `99.9%` | "ninety-nine point nine percent" |
| `0.5%` | "zero point five percent" or "half a percent" |

### SSML Example

```xml
<say-as interpret-as="unit">50%</say-as>
<!-- Output: "fifty percent" -->
```

---

## Ordinal Numbers

### Standard Ordinals

| Written | Spoken |
|---------|--------|
| `1st` | "first" |
| `2nd` | "second" |
| `3rd` | "third" |
| `4th` | "fourth" |
| `21st` | "twenty-first" |
| `100th` | "one hundredth" |

### SSML Example

```xml
<say-as interpret-as="ordinal">21</say-as>
<!-- Output: "twenty-first" -->
```

### Common Use Cases

- Rankings: "She finished in 3rd place" → "She finished in third place"
- Dates: "March 21st" → "March twenty-first"
- Floors: "Go to the 15th floor" → "Go to the fifteenth floor"

---

## Abbreviations & Acronyms

### Decision Matrix

| Type | How to Handle | Example |
|------|--------------|---------|
| **Spell out** | Add periods/spaces | `FBI` → "F.B.I." or "F B I" |
| **Read as word** | Common acronyms | `NATO` → "NATO" (reads as word) |
| **Expand fully** | For clarity | `SW` → "southwest" |

### Common Expansions

| Abbreviation | Expansion |
|--------------|-----------|
| `St.` | "Street" or "Saint" (context-dependent) |
| `Dr.` | "Doctor" or "Drive" |
| `etc.` | "et cetera" |
| `i.e.` | "that is" |
| `e.g.` | "for example" |
| `vs.` | "versus" |

### Best Practice for Voice Agents

- Maintain a pronunciation dictionary for domain-specific terms
- Use phonetic spellings for proper nouns: "Siobhan" → "shi-VAWN"

---

## Units of Measurement

### Expand All Abbreviated Units

| Abbreviation | Expansion |
|--------------|-----------|
| `kg` | "kilograms" |
| `cm` | "centimeters" |
| `m/s` | "meters per second" |
| `°C` | "degrees Celsius" |
| `°F` | "degrees Fahrenheit" |
| `mph` | "miles per hour" |
| `km/h` | "kilometers per hour" |
| `sq ft` | "square feet" |
| `m²` | "square meters" |

### Examples

| Input | Output |
|-------|--------|
| `5'10"` | "five feet ten inches" |
| `25°C` | "twenty-five degrees Celsius" |
| `100 km/h` | "one hundred kilometers per hour" |
| `3.5 kg` | "three point five kilograms" |

---

## Special Characters

### Character Replacement

| Character | Say As |
|-----------|--------|
| `&` | "and" |
| `*` | "asterisk" or "star" |
| `#` | "hash" or "number" |
| `+` | "plus" |
| `-` | "minus" or "dash" (context) |
| `=` | "equals" |
| `/` | "slash" or "divided by" |
| `\` | "backslash" |
| `|` | "pipe" or "bar" |
| `~` | "tilde" |
| `@` | "at" |

### Context-Dependent Examples

| Input | Context | Output |
|-------|---------|--------|
| `-5` | Temperature | "minus five" |
| `555-1234` | Phone | "five five five, one two three four" |
| `A-B` | Range | "A to B" or "A through B" |

---

## Credit Card Numbers

### Best Practices

- **Always read digit-by-digit** for clarity and security
- Group in fours with pauses
- Never read as a single large number

### Example

| Input | Output |
|-------|--------|
| `4111 1111 1111 1111` | "four one one one, one one one one, one one one one, one one one one" |

### Expiration Dates

| Input | Output |
|-------|--------|
| `12/25` | "December twenty twenty-five" or "twelve twenty-five" |
| `03/24` | "March twenty twenty-four" |

---

## SSML Reference

### Common `say-as` interpret-as Values

| Value | Use Case | Example |
|-------|----------|---------|
| `cardinal` | Regular numbers | "123" → "one hundred twenty-three" |
| `ordinal` | Ranked positions | "3" → "third" |
| `digits` | Digit by digit | "123" → "one two three" |
| `fraction` | Fractions | "3/4" → "three quarters" |
| `unit` | With units | "10%" → "ten percent" |
| `date` | Dates | With format attribute |
| `time` | Times | With format attribute |
| `telephone` | Phone numbers | Reading style varies |
| `currency` | Money | With currency code |
| `verbatim` / `spell-out` | Letter by letter | "ABC" → "A B C" |
| `characters` | Character by character | Including punctuation |

### SSML Quick Reference

```xml
<!-- Pause -->
<break time="500ms"/>

<!-- Emphasis -->
<emphasis level="strong">important</emphasis>

<!-- Pronunciation -->
<phoneme alphabet="ipa" ph="təˈmeɪtoʊ">tomato</phoneme>

<!-- Rate/Pitch/Volume -->
<prosody rate="slow" pitch="+10%" volume="loud">Slow and loud</prosody>

<!-- Say-as wrapper -->
<say-as interpret-as="date" format="mdy">12/25/2025</say-as>
```

---

## Implementation Recommendations

### 1. Create a Text Normalization Pipeline

```python
class TTSNormalizer:
    """Normalize text before sending to TTS engine."""
    
    def normalize(self, text: str, context: str = None) -> str:
        """
        Apply all normalization rules.
        
        Args:
            text: Raw text to normalize
            context: Optional context hint (e.g., "phone", "email", "currency")
        
        Returns:
            Normalized text ready for TTS
        """
        text = self.normalize_dates(text)
        text = self.normalize_times(text)
        text = self.normalize_numbers(text)
        text = self.normalize_currency(text)
        text = self.normalize_phone_numbers(text)
        text = self.normalize_emails(text)
        text = self.normalize_urls(text)
        text = self.normalize_units(text)
        text = self.normalize_abbreviations(text)
        text = self.normalize_special_characters(text)
        return text
```

### 2. Maintain a Pronunciation Dictionary

```json
{
  "proper_nouns": {
    "Siobhan": "shi-VAWN",
    "Nguyen": "win",
    "São Paulo": "sow PAO-lo"
  },
  "brand_names": {
    "iOS": "eye oh ess",
    "WiFi": "why fye"
  },
  "acronyms": {
    "API": "A P I",
    "URL": "U R L",
    "SQL": "sequel"
  }
}
```

### 3. Use SSML When Supported

If your TTS engine supports SSML, wrap structured data appropriately:

```python
def wrap_ssml(text: str, interpret_as: str, format: str = None) -> str:
    if format:
        return f'<say-as interpret-as="{interpret_as}" format="{format}">{text}</say-as>'
    return f'<say-as interpret-as="{interpret_as}">{text}</say-as>'
```

### 4. Test with Real Voices

- Different TTS engines handle normalization differently
- Always test with your production TTS voice
- Document edge cases specific to your TTS provider

### 5. Consider Locale/Language

| Consideration | Example |
|---------------|---------|
| Number formats | US: `1,234.56` vs BR: `1.234,56` |
| Date formats | US: MM/DD/YYYY vs EU: DD/MM/YYYY |
| Currency | Position of symbol varies |
| Decimal separator | Period vs comma |

---

## Provider-Specific Notes

### Cartesia

- Modern neural TTS with good built-in normalization
- Test specific edge cases with your voice

### ElevenLabs

- Has built-in text normalization
- Supports phonetic pronunciation overrides
- May omit numbers in sequences (test long IDs)

### OpenAI TTS

- Good general normalization
- Use prompts to guide pronunciation when needed

### Google Cloud TTS

- Full SSML support
- Extensive `say-as` options
- Good multilingual handling

### Amazon Polly

- Full SSML support
- Neural and standard voices differ in handling
- Lexicons for custom pronunciations

---

## Quick Cheat Sheet

| Data Type | Key Rule |
|-----------|----------|
| **Dates** | Write out fully: "October 12, 2025" |
| **Times** | Use natural language: "nine thirty p.m." |
| **Numbers** | Large = words, sequences = spaces |
| **Currency** | "$10.50" → "ten dollars and fifty cents" |
| **Phone** | Space/comma for pauses: "1 2 3, 4 5 6" |
| **Email** | Replace symbols: "at", "dot", "underscore" |
| **URLs** | Describe: "example dot com slash contact" |
| **Fractions** | Natural: "three quarters" |
| **Decimals** | Digit by digit after point: "three point one four" |
| **Ordinals** | "1st" → "first" |
| **Acronyms** | Periods or expand: "F.B.I." or "Federal Bureau" |
| **Units** | Always expand: "km" → "kilometers" |

---

*Last updated: December 2025*
