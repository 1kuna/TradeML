# GPT Pro Form 4 Historical Retrieval Path

Captured: 2026-05-05

The following is copied from the ChatGPT thread context. It is preserved as a
source artifact for iterative review.

---

Use a **hybrid manifest + raw XML path**. Do **not** rely only on issuer-level
submissions JSON for historical discovery, because some ownership filing archive
paths use a reporting-owner/filer CIK rather than the issuer CIK. Your parser
should treat the archive-path CIK as separate from `issuerCik`.

The robust path:

```text
SEC full-index / daily-index  -> canonical accession + archive path CIK
EDGAR submissions JSON        -> enrichment + acceptanceDateTime + primaryDocument when available
SEC Archives raw XML          -> authoritative parsed filing body
Complete .txt fallback        -> SGML header + XML extraction when raw-primary lookup fails
```

SEC's `data.sec.gov/submissions/CIK##########.json` gives company filing history
and recent filing metadata, with older submission files listed under `files`;
SEC says these APIs need no API key and are updated throughout the day as
submissions are disseminated. But SEC's archive/index docs are still the better
coverage backbone because the EDGAR indexes include the filing path/filename,
and archive paths can differ from the issuer CIK. ([SEC][1])

---

# 1. Historical retrieval path

## Primary historical manifest

For backtest history, build the manifest from SEC full indexes, not current
tickers.

Use quarterly full-index files:

```python
https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx
https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/form.idx
```

Filter exact form type:

```text
4
4/A
```

Start at **2006Q1** for this XML MVP. SEC's own insider transaction dataset
scope begins with Ownership XML submissions from January 2006, which is a
reasonable cutoff for avoiding old HTML/PEM-style Form 4 junk.

Each index row gives:

```text
CIK | Company Name | Form Type | Date Filed | Filename
```

The `Filename` is the canonical archive `.txt` path, for example:

```python
edgar/data/{archive_cik}/{accession}.txt
```

SEC explicitly documents that EDGAR indexes list company name, form type, CIK,
date filed, and filename including the folder path. ([SEC][2])

## Why full-index first

Because this breaks naive issuer-CIK retrieval.

Example: a Tiptree Form 4 has archive path CIK `769993`, while the issuer CIK
inside the filing is `1393726`. If you construct the archive URL from issuer CIK
only, you can miss or 404 some filings. The filing also has multiple reporting
owners and mixed `P`/`S` rows, making it a perfect fixture. ([SEC][3])

So store both:

```text
archive_cik    # from full-index filename path; used for URL construction
issuer_cik     # from ownership XML; used for company/entity mapping
owner_cik      # from ownership XML; used for insider identity
```

Do **not** use the accession prefix as the archive CIK. The accession prefix may
be a filing agent or submitter. SEC's docs note that the accession number's first
block is the CIK of the submitting entity, which may be the company or a
third-party filing agent. ([SEC][2])

---

# 2. Exact URL construction

Given:

```python
archive_cik = "769993"                 # from index filename path
accession = "0000769993-15-000534"
primary_document = "ownershipdoc03152015012806.xml"
```

Construct:

```python
from urllib.parse import quote

def sec_archive_dir(archive_cik: str | int, accession: str) -> str:
    cik_no_leading_zeros = str(int(str(archive_cik)))
    accession_no_dashes = accession.replace("-", "")
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_leading_zeros}/{accession_no_dashes}/"
    )

def raw_primary_xml_url(archive_cik: str | int, accession: str, primary_document: str) -> str:
    return sec_archive_dir(archive_cik, accession) + quote(primary_document)

def complete_txt_url_from_index_filename(index_filename: str) -> str:
    return "https://www.sec.gov/Archives/" + index_filename

def complete_txt_url_from_parts(archive_cik: str | int, accession: str) -> str:
    cik_no_leading_zeros = str(int(str(archive_cik)))
    return f"https://www.sec.gov/Archives/edgar/data/{cik_no_leading_zeros}/{accession}.txt"

def accession_index_url(archive_cik: str | int, accession: str) -> str:
    return sec_archive_dir(archive_cik, accession) + f"{accession}-index.htm"

def accession_directory_index_json_url(archive_cik: str | int, accession: str) -> str:
    return sec_archive_dir(archive_cik, accession) + "index.json"

def sgml_header_url(archive_cik: str | int, accession: str) -> str:
    return sec_archive_dir(archive_cik, accession) + f"{accession}.hdr.sgml"
```

Example raw XML URL:

```python
raw_primary_xml_url(
    archive_cik="1971213",
    accession="0001250842-25-000026",
    primary_document="primary_doc.xml",
)
# https://www.sec.gov/Archives/edgar/data/1971213/000125084225000026/primary_doc.xml
```

Important rule:

```text
Never build from:
  accession prefix
  issuerCik from XML
  issuerTradingSymbol
  company_tickers.json CIK

Always build from:
  archive_cik from SEC index filename path
  OR a verified archive URL already returned by SEC
```

Also: do **not** scrape the rendered `xslF345X03/` or `xslF345X05/` URL. That
is the browser-rendered ownership form route. Fetch the raw XML file in the
accession directory.

SEC documents the post-EDGAR 7.0 archive pattern as:

```python
/Archives/edgar/data/{cik}/{accession_without_dashes}/{accession}.txt
```

and says the accession-number directory contains all documents submitted for
that filing. ([SEC][2])

---

# 3. Retrieval algorithm

## Step A - Build accession manifest

Use `master.idx` or `form.idx`:

```python
{
    "archive_cik": "769993",
    "form": "4",
    "filed_date": "2015-04-15",
    "index_filename": "edgar/data/769993/0000769993-15-000534.txt",
    "accession": "0000769993-15-000534",
    "accession_no_dashes": "000076999315000534",
    "discovery_source": "sec_full_index",
    "index_year": 2015,
    "index_quarter": 2,
    "index_file_hash": "...",
    "index_crawled_at": "..."
}
```

## Step B - Enrich from submissions JSON where available

For issuer CIKs you already know, fetch:

```python
https://data.sec.gov/submissions/CIK{issuer_cik_10_digits}.json
```

Then fetch older submission files listed in:

```python
root["files"][i]["name"]
```

using:

```python
https://data.sec.gov/submissions/{name}
```

Use submissions JSON to enrich:

```text
acceptanceDateTime
primaryDocument
primaryDocDescription
reportDate
filmNumber
fileNumber
isXBRL / isInlineXBRL
```

But do not trust submissions JSON alone for archive path construction unless the
raw XML URL is verified.

## Step C - Get primary XML

Preferred path:

```python
GET raw_primary_xml_url(archive_cik, accession, primaryDocument)
```

Fallbacks:

1. Fetch accession index page and parse the document table for the document
   whose type is `4` or `4/A`.
2. Fetch complete `.txt` and parse the SGML document blocks.
3. Extract the `<XML>...</XML>` block for the ownership primary document.

Pseudo:

```python
def extract_primary_ownership_xml_from_complete_txt(txt: str) -> str | None:
    docs = split_sgml_documents(txt)  # split on <DOCUMENT>...</DOCUMENT>

    for doc in docs:
        doc_type = sgml_tag(doc, "TYPE")
        filename = sgml_tag(doc, "FILENAME")

        if doc_type in {"4", "4/A"}:
            xml = between(doc, "<XML>", "</XML>")
            if xml and "<ownershipDocument" in xml:
                return xml

            # Some old files may have XML-ish body without clean XML wrapper.
            if filename and filename.lower().endswith(".xml") and "<ownershipDocument" in doc:
                return doc[doc.find("<ownershipDocument"):]

    return None
```

Store all fallbacks and failures:

```text
primary_xml_source = raw_primary_url | complete_txt_xml_block | failed
raw_xml_sha256
complete_txt_sha256
parse_status
parse_error
```

## Step D - Parse accepted timestamp

Use, in order:

```text
submissions.acceptanceDateTime
SGML <ACCEPTANCE-DATETIME>
accession index Accepted value
```

Store the raw string. Normalize carefully.

For trading labels, treat SEC accepted times as **America/New_York market
clock** unless your parser proves the API value includes a real timezone offset.
SEC documents EDGAR business hours in ET and notes dissemination/index timing in
ET. ([SEC][2])

```python
{
    "accepted_at_raw": "20200403164253",
    "accepted_at_et": "2020-04-03 16:42:53 America/New_York",
    "accepted_at_utc": "...",
    "accepted_source": "sgml_header"
}
```

## Step E - Rate limits and headers

Set a real user agent and throttle globally. SEC's current fair-access guidance
says max request rate is 10 requests/second and asks scripted users to declare a
user agent. ([SEC][2])

Use:

```python
HEADERS = {
    "User-Agent": "YourProjectName your-email@example.com",
    "Accept-Encoding": "gzip, deflate",
}
```

Run at 3-5 req/sec, not 10, unless you enjoy being blocked.

---

# 4. Amendment handling for `4/A`

Treat `4/A` as a **replacement/correction disclosure**, not a fresh ordinary buy
signal.

SEC's insider flat-file docs explicitly include Form 4/A as "Amendment of a
previous Form 4," and the XML data is "as filed," including amendments,
redundancies, inconsistencies, and discrepancies.

## Parse these fields

From the root:

```text
documentType
periodOfReport
dateOfOriginalSubmission
notSubjectToSection16
```

From transactions:

```text
transactionFormType
transactionCode
transactionDate
transactionShares
transactionPricePerShare
transactionAcquiredDisposedCode
```

Important: in a `documentType == 4/A` filing, transaction rows often still have:

```xml
<transactionFormType>4</transactionFormType>
```

That is normal. Do **not** require `transactionFormType == "4/A"`. The ownership
XML spec says transaction form type is mandatory and must be `4` or `5`; the
amendment status is at the document/submission level. ([SEC][4])

## Link amendment to original

Use this order:

```text
1. Exact original accession if available from your own mapping.
2. issuer_cik + owner_cik_set + dateOfOriginalSubmission.
3. issuer_cik + owner_cik_set + periodOfReport + overlapping transaction dates.
4. issuer_cik + owner_cik_set + similar row hash within +/-30 calendar days.
```

Canonical row hash for comparison:

```python
row_hash_fields = [
    "issuer_cik",
    "owner_cik_set",
    "security_title_norm",
    "transaction_date",
    "transaction_form_type",
    "transaction_code",
    "acquired_disposed",
    "shares_decimal",
    "price_decimal",
    "direct_or_indirect",
    "post_shares_decimal",
]
```

## Primary MVP policy

Use this:

```text
documentType == "4":
  eligible for primary signal if strict P-buy rules pass.

documentType == "4/A":
  parse and store.
  link to original.
  exclude from primary backtest signal.
```

Then add a secondary research bucket:

```text
amendment_revealed_new_buy = true
```

only if:

```text
original Form 4 is missing
OR original had no P/A common-stock buy row
OR amendment materially changes code/shares/price/acquired-disposed
```

But even then, label it separately:

```text
event_type = FORM4A_CORRECTION_DISCLOSED_BUY
```

Do not mix `4/A` with primary `FORM4_OPEN_MARKET_BUY`. Amendments are a
different event class.

---

# 5. XML parser requirements

Use namespace-agnostic XML parsing. EDGAR ownership XML usually has no painful
namespaces, but write it defensively.

## Required XPath-ish fields

```text
/ownershipDocument/schemaVersion
/ownershipDocument/documentType
/ownershipDocument/periodOfReport
/ownershipDocument/dateOfOriginalSubmission
/ownershipDocument/notSubjectToSection16

/ownershipDocument/issuer/issuerCik
/ownershipDocument/issuer/issuerName
/ownershipDocument/issuer/issuerTradingSymbol

/ownershipDocument/reportingOwner/reportingOwnerId/rptOwnerCik
/ownershipDocument/reportingOwner/reportingOwnerId/rptOwnerName
/ownershipDocument/reportingOwner/reportingOwnerRelationship/isDirector
/ownershipDocument/reportingOwner/reportingOwnerRelationship/isOfficer
/ownershipDocument/reportingOwner/reportingOwnerRelationship/isTenPercentOwner
/ownershipDocument/reportingOwner/reportingOwnerRelationship/isOther
/ownershipDocument/reportingOwner/reportingOwnerRelationship/officerTitle

/ownershipDocument/nonDerivativeTable/nonDerivativeTransaction/securityTitle/value
/ownershipDocument/nonDerivativeTable/nonDerivativeTransaction/transactionDate/value
/ownershipDocument/nonDerivativeTable/nonDerivativeTransaction/deemedExecutionDate/value
/ownershipDocument/nonDerivativeTable/nonDerivativeTransaction/transactionCoding/transactionFormType
/ownershipDocument/nonDerivativeTable/nonDerivativeTransaction/transactionCoding/transactionCode
/ownershipDocument/nonDerivativeTable/nonDerivativeTransaction/transactionCoding/equitySwapInvolved
/ownershipDocument/nonDerivativeTable/nonDerivativeTransaction/transactionAmounts/transactionShares/value
/ownershipDocument/nonDerivativeTable/nonDerivativeTransaction/transactionAmounts/transactionPricePerShare/value
/ownershipDocument/nonDerivativeTable/nonDerivativeTransaction/transactionAmounts/transactionAcquiredDisposedCode/value
/ownershipDocument/nonDerivativeTable/nonDerivativeTransaction/postTransactionAmounts/sharesOwnedFollowingTransaction/value
/ownershipDocument/nonDerivativeTable/nonDerivativeTransaction/ownershipNature/directOrIndirectOwnership/value
/ownershipDocument/nonDerivativeTable/nonDerivativeTransaction/ownershipNature/natureOfOwnership/value

/ownershipDocument/footnotes/footnote[@id]
/ownershipDocument/ownerSignature/signatureName
/ownershipDocument/ownerSignature/signatureDate
/ownershipDocument/remarks
```

SEC's ownership XML spec maps these non-derivative transaction elements to Form
4 Table I columns, including security title, transaction date, transaction code,
shares, price, acquired/disposed code, post-transaction shares, and ownership
form. ([SEC][4])

---

# 6. Field edge cases you must handle

## `P` does not mean "open-market" only

SEC defines code `P` as **open market or private purchase** of a non-derivative
or derivative security. ([SEC][5])

For an "open-market insider-buy" MVP, your strict inclusion should be:

```text
documentType == 4
table == nonDerivativeTransaction
transactionCode == P
transactionAcquiredDisposedCode == A
security_title passes common-stock filter
transactionShares > 0
transactionPricePerShare > 0
not private-placement/unit/SPAC-sponsor/PIPE flagged
not derivative table
```

Add private-placement exclusion flags:

```python
private_purchase_terms = [
    "private placement",
    "purchase agreement",
    "subscription agreement",
    "PIPE",
    "unit",
    "units",
    "sponsor",
    "initial public offering",
    "simultaneously with the consummation",
    "warrant",
]
```

If those appear in footnotes/remarks/security titles, mark:

```text
probably_private_or_unit_purchase = true
```

and exclude from the strict open-market MVP.

## Boolean variants

Parse all of these:

```text
0 / 1
true / false
True / False
TRUE / FALSE
```

Fields affected:

```text
isDirector
isOfficer
isTenPercentOwner
isOther
equitySwapInvolved
notSubjectToSection16
```

## Dates with timezone suffixes

You will see transaction dates like:

```text
2015-01-26
2015-01-26-05:00
```

For `transactionDate`, parse as a **date**, not UTC midnight. Keep the raw
string.

## Decimal precision

Do not cast to float. Use `Decimal`.

You will see prices like:

```text
1922.6925
14.9807
7.3901
```

The Amazon fixture has many weighted-average price rows with 4+ decimal places.
([SEC][6])

## Footnotes are field-level, not just filing-level

Every `<value>` wrapper can have multiple `footnoteId`s. Store per-field
footnote references:

```python
{
    "transaction_price": Decimal("1922.6925"),
    "transaction_price_footnotes": ["F2"],
    "security_title_footnotes": [],
    "transaction_date_footnotes": [],
}
```

Do not just concatenate footnotes into remarks.

The ownership XML spec explicitly allows multiple fields to reference the same
footnote, and multiple footnotes can attach to one field. ([SEC][4])

## Multiple reporting owners

Do not assume one owner. Do not assume the first owner is "the insider."

SEC's XML spec says the order of multiple reporting owners is not important and
there is no primary owner. ([SEC][4])

Represent:

```python
owner_cik_set = tuple(sorted(owner_ciks))
n_reporting_owners = len(owner_cik_set)
```

## Multiple signatures

Ignore for event ownership. SEC says signatures are not tied to reporting owners
and signature count need not match reporting-owner count. ([SEC][4])

## Supporting documents

Ownership submissions have one primary XML document and may have supporting
docs. The primary document must end with `.xml` and conform to the ownership
schema; supporting docs can be `.txt`, `.htm`, `.jpg`, `.gif`. ([SEC][4])

So:

```text
parse primary XML only
inventory supporting docs
ignore EX-24 / EX-99 / GRAPHIC for alpha extraction
```

But keep their presence as telemetry:

```text
public_document_count
has_ex99
has_ex24
has_graphic
```

## Security title normalization

Create both raw and normalized fields.

```python
COMMON_LIKE_ALLOW = [
    "common stock",
    "class a common stock",
    "class b common stock",
    "ordinary shares",
    "ordinary share",
    "common shares",
    "common share",
]

EXCLUDE_TERMS = [
    "warrant",
    "option",
    "restricted stock unit",
    "rsu",
    "preferred",
    "convertible",
    "unit",
    "right to receive",
    "phantom",
    "performance stock unit",
    "depositary share",
    "common units",
    "limited partnership",
]
```

Strict MVP:

```text
include common stock / ordinary shares
exclude units, warrants, preferred, RSUs, options, derivative table
```

## Price missing or zero

Even if `P/A`, exclude if:

```text
transactionPricePerShare is null
transactionPricePerShare <= 0
```

For strict open-market buys, price missing usually means you are not looking at
a clean market purchase.

## Mixed P/S same filing

Do not net unless you explicitly build a net-insider-flow feature.

For MVP:

```text
keep P/A rows
flag same_filing_has_sales = true
flag same_owner_same_day_has_sales = true
```

Consider excluding mixed P/S filings from the strict primary test, then include
them in a secondary bucket.

---

# 7. Event eligibility after parsing

Strict MVP eligibility:

```python
eligible = (
    document_type == "4"
    and table == "nonDerivativeTransaction"
    and transaction_code == "P"
    and acquired_disposed == "A"
    and transaction_form_type == "4"
    and shares is not None and shares > 0
    and price is not None and price > 0
    and common_stock_filter_pass
    and not probably_private_or_unit_purchase
    and not not_subject_to_section16
)
```

At issuer-event aggregation:

```text
event timestamp = accession accepted_at
event identity = issuer_cik + accepted_at + accession
duplicate suppression = same issuer, same owner set, same transaction dates within 5 trading days
```

For amendments:

```python
if document_type == "4/A":
    parse = True
    primary_signal_eligible = False
    amendment_bucket_eligible = maybe
```

---

# 8. Minimum fixture set before backtesting

Run these before any research result is trusted.

| Fixture | Accession | Why it matters | Expected assertion |
| --- | ---: | --- | --- |
| Amazon / Indra Nooyi | `0001127602-20-013168` | Many `P/A` rows, weighted-average price footnotes, indirect trust ownership, 10b5-1 footnote. ([SEC][6]) | Parser preserves all rows, Decimal prices, field-level footnotes, and aggregates purchase value correctly. |
| Sinclair / David Smith | `0001250842-25-000026` | Class A common stock, multiple transaction dates, weighted-average footnotes, large buy. ([SEC][7]) | Common-stock filter accepts Class A; event timestamp is filing acceptance, not transaction date. |
| Tiptree / Goldman | `0000769993-15-000534` | Archive CIK differs from issuer CIK; multiple reporting owners; mixed `P` and `S`; timezone-suffixed transaction dates. ([SEC][3]) | URL construction uses archive CIK from index; parser separates P buys from S sales; no owner-order assumption. |
| Immediatek / Radical / Mark Cuban | `0001209191-06-060213` | `4/A`, multiple reporting owners, EX-99 supporting document, OTC ticker, joint filer info. ([SEC][8]) | Parse but exclude from primary MVP because amendment + OTC; inventory supporting docs without parsing them as transactions. |
| Super Micro / Sara Liu | `0001758554-19-000046` | `4/A`, public doc count 3, graphics, mechanical `M` and `F` rows, holdings rows. ([SEC][9]) | No P-buy event generated; rows remain usable for negative-control infrastructure. |
| Archimedes Tech SPAC | `0001437749-25-003569` | Ordinary-share `P` plus derivative warrant `P`; private units/sponsor language; price blanks. ([SEC][10]) | Strict open-market filter excludes private/unit/SPAC sponsor filing; derivative P rows ignored. |
| Bioject / Logomasini | `0000810084-13-000003` | Derivative warrants/preferred with `P`, zero prices, underlying common references. ([SEC][11]) | No false common-stock buy from derivative table; zero-price rows excluded. |
| Eledon / private placement-style filing | `0001593968-24-000563` | Common stock `P` with accompanying warrant purchases and private-placement language. ([SEC][12]) | Private-placement flag triggers; exclude from strict open-market sample. |
| Ares / Ressler | `0001025978-25-000011` | Sales-only, 10b5-1, many weighted-average sale rows. ([SEC][13]) | No buy event; valid negative-control/sales fixture. |
| Very late filing | `0001528597-26-000004` | Transaction period long before filing date. ([SEC][14]) | `days_since_transaction` computed; exclude or bucket separately if beyond threshold. |

Minimum pass/fail tests:

```text
1. Raw XML retrieval works from archive_cik + accession + primaryDocument.
2. Tiptree fails if issuer_cik is wrongly used for archive path.
3. 4/A filings parse but do not create primary signals.
4. Derivative P rows do not become common-stock buy events.
5. Private placement/unit/SPAC P rows are flagged/excluded.
6. Weighted-average footnotes are preserved.
7. Decimal prices are not rounded.
8. Multiple reporting owners become an owner set, not one owner.
9. Mixed P/S filings are flagged.
10. Transaction date timezone suffixes parse without shifting the date.
```

---

# 9. Source-quality telemetry to log

For each accession:

```python
{
    "accession": str,
    "archive_cik": str,
    "index_filename": str,
    "manifest_source": "full_index" | "submissions" | "daily_index",
    "form_from_index": "4" | "4/A",
    "document_type_from_xml": "4" | "4/A",
    "filed_date_from_index": date,
    "accepted_at_raw": str | None,
    "accepted_at_source": "submissions" | "sgml" | "index_html" | None,

    "primary_document": str | None,
    "primary_xml_url": str | None,
    "primary_xml_http_status": int | None,
    "primary_xml_sha256": str | None,

    "complete_txt_url": str,
    "complete_txt_http_status": int | None,
    "complete_txt_sha256": str | None,

    "xml_source": "raw_primary" | "complete_txt_extracted" | "failed",
    "parse_status": "ok" | "xml_error" | "missing_ownershipDocument" | "no_primary_xml",
    "parse_error": str | None,

    "issuer_cik": str | None,
    "issuer_symbol_raw": str | None,
    "owner_cik_count": int,
    "n_nonderiv_transactions": int,
    "n_deriv_transactions": int,
    "n_p_a_common_candidates": int,

    "quality_flags": list[str],
}
```

Quality flags:

```text
archive_cik_differs_from_issuer_cik
accession_prefix_differs_from_archive_cik
raw_primary_404
used_complete_txt_fallback
document_type_mismatch
amendment
missing_date_of_original_submission
multiple_reporting_owners
supporting_documents_present
graphic_documents_present
mixed_p_and_s
derivative_p_present
private_or_unit_purchase_flag
missing_price
zero_price
non_common_security_title
otc_symbol
late_report
```

---

# 10. Concrete storage layout

Since you already have NAS parquet raw/curated:

```text
raw/sec/form4_manifest/year=YYYY/qtr=Q/
  manifest.parquet

raw/sec/archives/archive_cik={archive_cik}/accession={accession_no_dashes}/
  primary.xml
  complete.txt
  header.sgml
  accession_index.html
  directory_index.json
  metadata.json

curated/sec/form4/submissions/
  document-level parsed rows

curated/sec/form4/reporting_owners/
  one row per owner per accession

curated/sec/form4/nonderiv_transactions/
  one row per nonDerivativeTransaction

curated/sec/form4/deriv_transactions/
  one row per derivativeTransaction

curated/sec/form4/footnotes/
  one row per accession + footnote_id

curated/events/form4_open_market_buy_candidates/
  issuer-event aggregated candidates
```

Do not overwrite amendments into originals. Store append-only, then build a
resolved view:

```text
raw accession rows: immutable
resolved amendment view: replacement logic
primary event view: excludes 4/A
```

---

# 11. The harsh bit

The biggest trap is calling every `P` an open-market insider buy. SEC's own code
definition says `P` includes **private purchases**, and the weird fixtures will
prove how much garbage slips in: SPAC sponsor units, warrants, private
placements, preferred stock, mixed broker/dealer churn, late amendments, and
derivative rows.

For the first backtest, the strictest version should be:

```text
Form 4 only
non-derivative only
P/A only
common stock / ordinary shares only
positive price
positive shares
no private-placement/unit/SPAC/warrant flags
no 4/A
no OTC
no mixed P/S primary sample
```

That may shrink the sample. Good. A small clean sample that falsifies honestly
is better than a big polluted sample that prints fake alpha.

[1]: https://www.sec.gov/search-filings/edgar-application-programming-interfaces "SEC.gov | EDGAR Application Programming Interfaces (APIs)"
[2]: https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data "SEC.gov | Accessing EDGAR Data"
[3]: https://www.sec.gov/Archives/edgar/data/769993/0000769993-15-000534.txt "www.sec.gov"
[4]: https://www.sec.gov/info/edgar/ownershipxmltechspec-v3_d.pdf "EDGAR Ownership XML Technical Specification"
[5]: https://www.sec.gov/edgar/searchedgar/ownershipformcodes.html "Ownership Form Codes"
[6]: https://www.sec.gov/Archives/edgar/data/1018724/0001127602-20-013168.txt "www.sec.gov"
[7]: https://www.sec.gov/Archives/edgar/data/1971213/000125084225000026/0001250842-25-000026.txt "www.sec.gov"
[8]: https://www.sec.gov/Archives/edgar/data/1084182/0001209191-06-060213.txt "www.sec.gov"
[9]: https://www.sec.gov/Archives/edgar/data/1375365/000175855419000046/0001758554-19-000046.txt "poa_saraliu"
[10]: https://www.sec.gov/Archives/edgar/data/2028516/000143774925003569/0001437749-25-003569.txt?utm_source=chatgpt.com "0001437749-25-003569.txt"
[11]: https://www.sec.gov/Archives/edgar/data/1480077/0000810084-13-000003.txt?utm_source=chatgpt.com "0000810084-13-000003.txt"
[12]: https://www.sec.gov/Archives/edgar/data/1824893/000159396824000563/0001593968-24-000563.txt?utm_source=chatgpt.com "0001593968-24-000563.txt"
[13]: https://www.sec.gov/Archives/edgar/data/1176948/000102597825000011/0001025978-25-000011.txt "www.sec.gov"
[14]: https://www.sec.gov/Archives/edgar/data/1569345/000152859726000004/0001528597-26-000004.txt?utm_source=chatgpt.com "0001528597-26-000004.txt"
