import streamlit as st
import pdfplumber
import re
import pandas as pd
import tempfile
from pdf2image import convert_from_path
from PIL import Image, ImageOps
import pytesseract

# Ścieżka do tesseract.exe (zmień, jeśli masz inną)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def extract_all_ids_from_gramatury(pdf_path):
    """
    Zwraca posortowaną listę wszystkich unikalnych ID znalezionych w pliku gramatury.
    """
    ids = set()
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            found = re.findall(r'\(ID:\s*(\d+)\)', text)
            ids.update(found)
    return sorted(ids, key=lambda x: int(x))

def autocrop_image(image):
    gray = image.convert("L")
    inverted = ImageOps.invert(gray)
    bbox = inverted.getbbox()
    if bbox:
        return image.crop(bbox)
    else:
        return image

def find_lp_y(page):
    for word in page.extract_words():
        if re.search(r'\bLp\.', word['text']):
            return word['top']
    return None

def extract_multiline_field(lines, field):
    value_lines = []
    capture = False
    field_pattern = re.compile(rf"^{field}:", re.IGNORECASE)
    next_fields = [
        "dieta:", "wariant:", "posiłek:", "zamówiono:", "zamówiono (wpot):"
    ]
    indices = []
    for idx, line in enumerate(lines):
        if field_pattern.match(line.strip()):
            value_lines.append(line.split(":", 1)[1].strip())
            capture = True
            indices.append(idx)
            continue
        if capture:
            if any(line.strip().lower().startswith(f) for f in next_fields):
                break
            value_lines.append(line.strip())
            indices.append(idx)
    return " ".join(value_lines).strip(), set(indices)

def extract_recipe_info(page, danie_id):
    text = page.extract_text()
    if not text or f"(ID: {danie_id})" not in text:
        return None

    lines = text.split('\n')
    tytul = lines[1] if len(lines) > 1 else "Brak tytułu"
    tytul = re.sub(r'^przepisy?\s*', '', tytul, flags=re.IGNORECASE)

    dieta = wariant = posilek = dania = zamowiono = zamowiono_wpot = ""
    dania_indices = set()
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith("dieta:"):
            dieta = line.split(":", 1)[1].strip()
        if line.strip().lower().startswith("wariant:"):
            wariant = line.split(":", 1)[1].strip()
        if line.strip().lower().startswith("posiłek:"):
            posilek = line.split(":", 1)[1].strip()
        if line.strip().lower().startswith("zamówiono (wpot):"):
            zamowiono_wpot = line.split(":", 1)[1].strip()
        elif line.strip().lower().startswith("zamówiono:"):
            zamowiono = line.split(":", 1)[1].strip()
    dania, dania_indices = extract_multiline_field(lines, "dania")

    opis_lines = []
    opis_start = 2
    for i in range(opis_start, len(lines)):
        if re.search(r'\bLp\.', lines[i]):
            break
        if i in dania_indices:
            continue
        if any([
            lines[i].strip().lower().startswith("dieta:"),
            lines[i].strip().lower().startswith("wariant:"),
            lines[i].strip().lower().startswith("posiłek:"),
            lines[i].strip().lower().startswith("zamówiono:"),
            lines[i].strip().lower().startswith("zamówiono (wpot):")
        ]):
            continue
        opis_lines.append(lines[i])
    opis = "\n".join(opis_lines).strip()

    skladniki_start = None
    skladniki_end = None
    for i, line in enumerate(lines):
        if re.match(r'\s*Składniki', line, re.IGNORECASE):
            skladniki_start = i
        if skladniki_start is not None and line.strip().lower().startswith("razem"):
            skladniki_end = i
            break

    df = None
    if skladniki_start is not None and skladniki_end is not None:
        skladniki_lines = lines[skladniki_start:skladniki_end]
        header = re.split(r'\s{2,}|\t', skladniki_lines[0].strip())
        data = []
        for row in skladniki_lines[1:]:
            cols = re.split(r'\s{2,}|\t', row.strip())
            cols += [""] * (len(header) - len(cols))
            data.append(cols)
        df = pd.DataFrame(data, columns=header)

    return {
        "tytul": tytul,
        "opis": opis,
        "skladniki": df,
        "dieta": dieta,
        "wariant": wariant,
        "posilek": posilek,
        "dania": dania,
        "zamowiono": zamowiono,
        "zamowiono_wpot": zamowiono_wpot
    }

def extract_gramatura_info(pdf, danie_id):
    for page in pdf.pages:
        text = page.extract_text()
        if not text:
            continue
        pattern = r'\(ID: ?' + str(danie_id) + r'\)'
        if re.search(pattern, text):
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if re.search(pattern, line):
                    match = re.search(r'\(ID: ?\d+\)\s*(.*)', line)
                    nazwa_dania = match.group(1).strip() if match else "Nazwa nieznana"
                    fragment = lines[i:i+5]
                    fragment = [l for l in fragment if not re.fullmatch(r'\d+\s*', l.strip()) and l.strip()]
                    return nazwa_dania, "\n".join(fragment)
    return "Nazwa nieznana", "Nie znaleziono gramatury dla tego ID."

def extract_gramatura_image_fragment_for_id(pdf_path, danie_id, margin=50):
    id_pattern = re.compile(rf"\b{re.escape(str(danie_id))}\b")
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue
            if not id_pattern.search(text):
                continue
            y_start = None
            y_end = None
            for word in page.extract_words():
                if y_start is None and re.search(r'Przepisy/Składniki', word['text'], re.IGNORECASE):
                    y_start = word['top']
                if re.search(r'Pudełka', word['text'], re.IGNORECASE):
                    y_end = word['bottom']
            if y_start is not None and y_end is not None:
                images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
                img = images[0]
                pdf_height = page.height
                img_height = img.height
                y_start_px = int(y_start * img_height / pdf_height) - margin
                y_end_px = int(y_end * img_height / pdf_height) + margin
                y_start_px = max(0, y_start_px)
                y_end_px = min(img.height, y_end_px)
                cropped_img = img.crop((0, y_start_px, img.width, y_end_px))
                return cropped_img
    return None

def extract_przepis_image_fragment_for_page(pdf_path, page_num, margin_top=20, margin_bottom=50):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        lp_y = find_lp_y(page)
        images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
        img = images[0]
        if lp_y is not None:
            pdf_height = page.height
            img_height = img.height
            y_px = int(lp_y * img_height / pdf_height) - margin_top
            y_px = max(0, y_px)
            bottom_px = min(img.height, img.height + margin_bottom)
            cropped_img = img.crop((0, y_px, img.width, bottom_px))
            cropped_img = autocrop_image(cropped_img)
            return cropped_img
        else:
            cropped_img = autocrop_image(img)
            return cropped_img

def extract_ids_from_dania(dania_text, main_id):
    found_ids = set(re.findall(r'\(ID:\s*(\d+)\)', dania_text))
    found_ids.discard(str(main_id))
    return list(found_ids)

def extract_ingredients_from_gramatura_image(image):
    text = pytesseract.image_to_string(image, lang='pol')
    lines = text.split('\n')
    ingredients = []
    blacklist = {'zamówiono', 'kt', 'ki', 'gramatury', 'gramatur', 'kcal'}
    substrings = ['office', 'standard', 'sport', 'wege', 'student', 'bez', 'pudełka']
    for line in lines:
        match = re.search(r'([a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ\s\-]+)\s+(\d+[.,]?\d*)\s*(kg|g)?', line)
        if match:
            name = match.group(1).strip()
            value = match.group(2).replace(',', '.')
            unit = match.group(3)
            name_lower = name.lower()
            if name_lower in blacklist:
                continue
            if any(sub in name_lower for sub in substrings):
                continue
            if len(name.replace(" ", "")) < 3:
                continue
            try:
                weight = float(value)
                if unit == 'kg':
                    weight = weight * 1000
                ingredients.append((name, weight))
            except ValueError:
                pass
    return ingredients

def calculate_pre_cooking_weights(ingredients, excel_df):
    results = []
    for name, weight_wpot in ingredients:
        wpot_percent = None
        for idx, row in excel_df.iterrows():
            excel_name = str(row['Nazwa']).lower() if pd.notnull(row['Nazwa']) else ""
            ingredient_name = name.lower()
            if (ingredient_name in excel_name or
                any(word in excel_name for word in ingredient_name.split() if len(word) > 3)):
                if len(row) >= 5 and pd.notnull(row.iloc[4]):
                    try:
                        wpot_percent = float(row.iloc[4])
                    except (ValueError, TypeError):
                        wpot_percent = None
                break
        if wpot_percent is not None:
            wpot_coef = wpot_percent / 100
            weight_before = weight_wpot / wpot_coef
            results.append({
                'name': name,
                'weight_wpot': weight_wpot,
                'weight_before': weight_before,
                'wpot_percent': wpot_percent,
                'found_coefficient': True
            })
        else:
            results.append({
                'name': name,
                'weight_wpot': weight_wpot,
                'found_coefficient': False
            })
    return results

def main():
    st.title("Master Przepis Generator (przepisy + gramatury + obraz od tabeli)")

    przepisy_file = st.file_uploader("Wgraj plik z przepisami (PDF)", type="pdf")
    gramatury_file = st.file_uploader("Wgraj plik z gramaturami (PDF)", type="pdf")
    skladniki_file = st.file_uploader("Wgraj plik ze składnikami (Excel)", type=["xlsx", "xls"])

    if przepisy_file and gramatury_file and skladniki_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_przepisy:
            tmp_przepisy.write(przepisy_file.read())
            tmp_przepisy_path = tmp_przepisy.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_gramatury:
            tmp_gramatury.write(gramatury_file.read())
            tmp_gramatury_path = tmp_gramatury.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_skladniki:
            tmp_skladniki.write(skladniki_file.read())
            tmp_skladniki_path = tmp_skladniki.name

        try:
            excel_df = pd.read_excel(tmp_skladniki_path)
        except Exception as e:
            st.error(f"Błąd wczytywania pliku Excel: {e}")
            return

        # Wyciągnij wszystkie ID z gramatury
        all_ids = extract_all_ids_from_gramatury(tmp_gramatury_path)

        # Iteruj po wszystkich ID i wyświetl informacje
        for selected_id in all_ids:
            with pdfplumber.open(tmp_przepisy_path) as pdf_przepisy:
                przepisy = []
                for page_num, page in enumerate(pdf_przepisy.pages):
                    info = extract_recipe_info(page, selected_id)
                    if info:
                        info['page_num'] = page_num
                        przepisy.append(info)

            with pdfplumber.open(tmp_gramatury_path) as pdf_gramatury:
                nazwa_dania, gramatura = extract_gramatura_info(pdf_gramatury, selected_id)
                dodatkowe_id = []
                if przepisy:
                    dania_text = przepisy[0]['dania']
                    dodatkowe_id = extract_ids_from_dania(dania_text, selected_id)
                dodatkowe_gramatury = []
                for did in dodatkowe_id:
                    nazwa, gram = extract_gramatura_info(pdf_gramatury, did)
                    dodatkowe_gramatury.append((did, nazwa, gram))

            st.header(f"Przepisy dla ID {selected_id}: {nazwa_dania}")
            if przepisy:
                for idx, przepis in enumerate(przepisy, 1):
                    st.markdown(f"## Przepis: {przepis['tytul']}")
                    st.markdown(f"**DIETA:** {przepis['dieta']}")
                    st.markdown(f"**WARIANT:** {przepis['wariant']}")
                    st.markdown(f"**POSIŁEK:** {przepis['posilek']}")
                    st.markdown(f"**DANIA:** {przepis['dania']}")
                    st.markdown(f"**ZAMÓWIONO:** {przepis['zamowiono']}")
                    st.markdown(f"**ZAMÓWIONO (WpOT):** {przepis['zamowiono_wpot']}")

                    if przepis['skladniki'] is not None and not przepis['skladniki'].empty:
                        st.markdown("### SKŁADNIKI DLA PRZEPISU:")
                        st.dataframe(przepis['skladniki'], use_container_width=True)
                    if przepis['opis']:
                        st.markdown("**Opis przygotowania:**")
                        st.write(przepis['opis'])

                    cropped_img = extract_przepis_image_fragment_for_page(
                        tmp_przepisy_path,
                        przepis['page_num'],
                        margin_top=20,
                        margin_bottom=50
                    )
                    st.markdown("**SKŁADNIKI DLA PRZEPISU**")
                    st.image(cropped_img, use_container_width=True)

            else:
                st.warning("Nie znaleziono przepisów dla tego ID.")

            st.header("Gramatura:")
            st.markdown(f"**{selected_id}: {nazwa_dania}**")
            st.text(gramatura)
            st.markdown("### Obraz fragmentu gramatury (od 'Przepisy/Składniki' do 'Pudełka') dla tego ID:")
            gramatura_img = extract_gramatura_image_fragment_for_id(tmp_gramatury_path, selected_id, margin=50)
            if gramatura_img is not None:
                st.image(gramatura_img, use_container_width=True)
                ingredients = extract_ingredients_from_gramatura_image(gramatura_img)
                if ingredients:
                    results = calculate_pre_cooking_weights(ingredients, excel_df)
                    st.markdown("### Przeliczone gramatury składników:")
                    for result in results:
                        if result['found_coefficient']:
                            st.markdown(f"**{result['name']}:**")
                            st.markdown(f"- Waga po obróbce (WpOT): {result['weight_wpot']:.0f} g")
                            st.markdown(f"- Waga przed obróbką: {result['weight_before']:.0f} g (przy WpOT = {result['wpot_percent']}%)")
                        else:
                            st.markdown(f"**{result['name']}:**")
                            st.markdown(f"- Waga po obróbce (WpOT): {result['weight_wpot']:.0f} g")
                            st.markdown(f"- Nie znaleziono współczynnika WpOT dla tego składnika")
                else:
                    st.warning("Nie udało się wyciągnąć składników z obrazu gramatury.")
            else:
                st.warning("Nie znaleziono fragmentu od 'Przepisy/Składniki' do 'Pudełka' dla tego ID w pliku gramatury.")

            for did, nazwa, gram in dodatkowe_gramatury:
                st.markdown(f"**{did}: {nazwa}**")
                st.text(gram)
                st.markdown(f"### Obraz fragmentu gramatury (od 'Przepisy/Składniki' do 'Pudełka') dla ID {did}:")
                gramatura_img = extract_gramatura_image_fragment_for_id(tmp_gramatury_path, did, margin=50)
                if gramatura_img is not None:
                    st.image(gramatura_img, use_container_width=True)
                    ingredients = extract_ingredients_from_gramatura_image(gramatura_img)
                    if ingredients:
                        results = calculate_pre_cooking_weights(ingredients, excel_df)
                        st.markdown("### Przeliczone gramatury składników:")
                        for result in results:
                            if result['found_coefficient']:
                                st.markdown(f"**{result['name']}:**")
                                st.markdown(f"- Waga po obróbce (WpOT): {result['weight_wpot']:.0f} g")
                                st.markdown(f"- Waga przed obróbką: {result['weight_before']:.0f} g (przy WpOT = {result['wpot_percent']}%)")
                            else:
                                st.markdown(f"**{result['name']}:**")
                                st.markdown(f"- Waga po obróbce (WpOT): {result['weight_wpot']:.0f} g")
                                st.markdown(f"- Nie znaleziono współczynnika WpOT dla tego składnika")
                    else:
                        st.warning("Nie udało się wyciągnąć składników z obrazu gramatury.")
                else:
                    st.warning(f"Nie znaleziono fragmentu od 'Przepisy/Składniki' do 'Pudełka' dla ID {did} w pliku gramatury.")

if __name__ == "__main__":
    main()
