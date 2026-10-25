import base64
import html
import os
import re

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# 1. КОНФІГУРАЦІЯ СТОРІНКИ
# ==========================================================
st.set_page_config(
    page_title="EcoVision AI — Autonomous Environmental Agent",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================================
# 2. ДИЗАЙН
# ==========================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }

    .stApp {
        background: linear-gradient(135deg, #022314 0%, #064E3B 40%, #0F172A 100%) !important;
        color: #FFFFFF !important;
    }

    html, body, [class*="css"], label, p, span, div, .stMarkdown p {
        color: #F8FAFC !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 19px !important;
        font-weight: 500 !important;
    }

    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, .stImage caption, div[class*="caption"] {
        color: #FFFFFF !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        text-align: center !important;
        background: rgba(15, 23, 42, 0.8) !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
        margin-top: 6px !important;
        border: 1px solid rgba(74, 222, 128, 0.4) !important;
    }

    [data-testid="stCameraInput"] button {
        background: linear-gradient(90deg, #15803D, #22C55E) !important;
        color: transparent !important;
        font-size: 0px !important;
        width: 100% !important;
        height: 52px !important;
        border-radius: 14px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.4) !important;
        position: relative !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stCameraInput"] button span, [data-testid="stCameraInput"] button svg { display: none !important; }
    [data-testid="stCameraInput"] button::after {
        content: "📷 Зробити знімок" !important;
        position: absolute !important; left: 0; right: 0; top: 0; bottom: 0;
        display: flex; align-items: center; justify-content: center;
        font-size: 18px; color: #FFFFFF; font-weight: 700;
        background: linear-gradient(90deg, #15803D, #22C55E); border-radius: 14px;
    }

    .info-card-bright {
        background: rgba(15, 23, 42, 0.85) !important;
        border-left: 5px solid #22C55E !important;
        border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 18px 24px !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
        font-size: 19px !important;
        line-height: 1.7 !important;
        margin-bottom: 20px !important;
    }

    .main-title {
        font-family: 'Inter', sans-serif !important;
        font-size: 46px !important;
        font-weight: 900 !important;
        background: linear-gradient(90deg, #00E676, #38BDF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px !important;
    }
    .subtitle {
        font-family: 'Inter', sans-serif !important;
        font-size: 20px !important;
        color: #A7F3D0 !important;
        margin-bottom: 25px !important;
        font-weight: 600 !important;
    }

    h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #4ADE80 !important;
    }

    @keyframes fadeInContent {
        0% { opacity: 0; transform: translateY(15px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .animated-section { animation: fadeInContent 0.4s ease-out forwards !important; }

    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 2px dashed #22C55E !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4) !important;
    }
    [data-testid="stFileUploader"] section { background: transparent !important; }
    [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] div { color: #FFFFFF !important; }

    [data-testid="stFileUploader"] button {
        background: linear-gradient(90deg, #15803D, #22C55E) !important;
        color: transparent !important;
        font-size: 0px !important;
        width: 170px !important;
        height: 48px !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.4) !important;
        position: relative !important;
    }
    [data-testid="stFileUploader"] button span, [data-testid="stFileUploader"] button svg { display: none !important; }
    [data-testid="stFileUploader"] button::after {
        content: "📁 Завантажити" !important;
        position: absolute !important; left: 0; right: 0; top: 0; bottom: 0;
        display: flex; align-items: center; justify-content: center;
        font-size: 17px; color: #FFFFFF; font-weight: 700;
        background: linear-gradient(90deg, #15803D, #22C55E); border-radius: 12px;
    }

    .chat-bubble-user {
        background: linear-gradient(135deg, #166534, #15803D);
        color: #FFFFFF; padding: 18px 24px;
        border-radius: 18px 18px 2px 18px; margin: 14px 0;
        font-size: 20px; font-weight: 500; max-width: 85%;
        margin-left: auto; box-shadow: 0 4px 15px rgba(21, 128, 61, 0.3);
    }
    .chat-bubble-agent {
        background: rgba(15, 23, 42, 0.95);
        border: 1px solid rgba(74, 222, 128, 0.5);
        color: #FFFFFF; padding: 24px 28px;
        border-radius: 18px 18px 18px 2px; margin: 14px 0;
        font-size: 20px; font-weight: 500; max-width: 92%;
        line-height: 1.85; box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }

    .stTextInput input {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        border: 2px solid #22C55E !important;
        height: 58px !important;
    }

    .stButton>button {
        width: 100% !important;
        border-radius: 14px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 800 !important;
        font-size: 18px !important;
        height: 58px !important;
        background: linear-gradient(90deg, #15803D, #22C55E) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(34, 197, 94, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #166534, #00E676) !important;
        box-shadow: 0 8px 30px rgba(34, 197, 94, 0.7) !important;
        transform: translateY(-3px) scale(1.01);
    }

    /* Повертаємо шрифт іконок Streamlit (глобальний шрифт їх ламав) */
    [data-testid="stIconMaterial"], span[class*="material-symbols"], span[class*="material-icons"] {
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
        font-weight: normal !important;
        font-size: 22px !important;
        letter-spacing: normal !important;
        font-feature-settings: 'liga' !important;
    }

    .heat-wrap { overflow-x: auto; }
    .heat-table { width: 100%; border-collapse: separate; border-spacing: 4px; margin: 10px 0 20px 0; }
    .heat-table th { color: #A7F3D0; font-size: 16px; font-weight: 700; text-align: center; padding: 6px; }
    .heat-table td.kind { text-align: left; font-weight: 700; font-size: 18px; padding: 10px 14px;
        background: rgba(15, 23, 42, 0.85); border-radius: 8px; white-space: nowrap; color: #FFFFFF; }
    .heat-table td.cell { text-align: center; font-weight: 800; font-size: 18px; padding: 12px 6px;
        border-radius: 8px; color: #FFFFFF; }
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# 3. ЯДРО ШІ: НАВЧЕНА НЕЙРОМЕРЕЖА (MobileNetV2, transfer learning)
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ecovision_model.keras")

# Порядок класів такий самий, як під час навчання (алфавітний)
CLASS_NAMES = ["battery", "biological", "cardboard", "glass",
               "metal", "paper", "plastic", "trash"]

CLASS_INFO = {
    "battery": ("🔋 Батарейка / акумулятор",
                "Спеціальний бокс для небезпечних відходів",
                "100+ років",
                "Містить важкі метали. Не викидайте у звичайний смітник."),
    "biological": ("🍏 Органіка (харчові відходи)",
                   "Коричневий контейнер (компост)",
                   "1–4 тижні",
                   "Біорозкладні залишки, придатні для компостування."),
    "cardboard": ("📦 Картон",
                  "Синій контейнер (папір і картон)",
                  "≈ 2–3 місяці",
                  "Чистий сухий картон підлягає переробці."),
    "glass": ("🍾 Скло",
              "Зелений контейнер (склотара)",
              "тисячі років",
              "Скло переробляється багаторазово без втрати якості."),
    "metal": ("🥫 Метал",
              "Жовтий контейнер (пластик і метал)",
              "50–200 років",
              "Металеву тару варто промити перед здаванням."),
    "paper": ("📄 Папір",
              "Синій контейнер (папір і картон)",
              "≈ 2–6 тижнів",
              "Чистий папір підлягає переробці."),
    "plastic": ("♻️ Пластик",
                "Жовтий контейнер (пластик і метал)",
                "100–500 років",
                "Промийте й сплюсніть тару перед сортуванням."),
    "trash": ("🗑️ Змішані відходи",
              "Загальний контейнер",
              "залежить від складу",
              "Не підлягає переробці на звичайних лініях."),
}


@st.cache_resource
def load_model():
    # Модель містить власний preprocess_input, тому на вхід йдуть пікселі 0..255
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


# ==========================================================
# 3.1 NLP-МОДУЛЬ: власна база знань + TF-IDF (char n-grams) + косинусна подібність
#     Без зовнішніх API. Додатково: контекст із CV та правила для класів відходів.
# ==========================================================
_decomp_lines = "\n".join(f"• {v[0]}: {v[2]}" for v in CLASS_INFO.values())

# (тема, [варіанти запитань], відповідь)
KB = [
    ("greeting", ["привіт", "добрий день", "вітаю", "доброго ранку", "hello", "хай", "добрий вечір"],
     "👋 **Вітаю! Я Еко-Агент EcoVision AI.**\nМожу розповісти про сортування відходів. "
     "А якщо завантажите фото у вкладці «Зір», підкажу, що з предметом робити далі."),
    ("thanks", ["дякую", "дуже дякую", "щиро дякую", "спасибі", "thanks", "дякую за допомогу"],
     "Радий допомогти! 🌱 Питайте ще, якщо щось цікавить."),
    ("bye", ["до побачення", "бувай", "па па", "на все добре", "до зустрічі"],
     "До зустрічі! Сортуйте відповідально ♻️"),
    ("about", ["хто ти", "що ти вмієш", "розкажи про себе", "що ти можеш", "для чого ти потрібен", "ти хто"],
     "Я Еко-Агент: поєдную **комп'ютерний зір** (розпізнаю тип відходу за фото) і **власний NLP-модуль** "
     "(відповідаю на питання про сортування). Можу враховувати те, що модель щойно побачила на фото."),
    ("how_use", ["як користуватися", "як працює додаток", "з чого почати", "як розпізнати відходи",
                 "як завантажити фото", "що робити в програмі"],
     "1) Відкрийте вкладку «Зір», завантажте фото або зробіть знімок.\n"
     "2) Натисніть «Запустити аналіз».\n"
     "3) Повертайтесь у чат і питайте, наприклад: «куди це викинути?»"),
    ("about_project", ["що це за проєкт", "для чого цей проєкт", "яку проблему вирішує проєкт",
                       "навіщо створено ecovision", "мета проєкту"],
     "EcoVision AI допомагає правильно сортувати побутові відходи. Неправильне сортування збільшує "
     "навантаження на полігони, а підказка «що це і в який контейнер» робить сортування простішим."),
    ("how_model", ["як працює модель", "яка нейромережа використовується", "на чому навчена модель",
                   "що таке mobilenet", "яка архітектура моделі", "як навчали модель"],
     "Це **MobileNetV2** з transfer learning: базова мережа з ImageNet і нова голова на 8 класів. "
     "Навчання дворівневе (спочатку голова, потім fine-tuning), з аугментацією й вагами класів. "
     "Дані: публічний датасет Garbage Classification, 8213 фото, поділ 70/15/15."),
    ("accuracy", ["яка точність моделі", "наскільки модель точна", "чи можна довіряти моделі",
                  "як часто модель помиляється", "скільки відсотків точності"],
     "На тестовій вибірці (1232 фото, яких модель не бачила) точність близько **92%**. "
     "На реальних фото з побуту вона нижча, бо датасет майже весь на однотонному фоні. "
     "Перевірити стійкість можна у вкладці «Стрес-тест»."),
    ("limits", ["чому модель помиляється", "модель плутає пластик і скло", "які обмеження системи",
                "коли модель не працює", "чому неправильний результат"],
     "Найчастіше модель плутає прозорі матеріали (пластик і скло) та блискучі предмети з металом. "
     "Гірше розпізнаються зім'яті, розбиті та нетипові предмети."),
    ("photo_tips", ["як краще сфотографувати", "поради для кращого фото", "який фон потрібен",
                    "чому низька впевненість", "як підвищити точність розпізнавання", "як сфотографувати щоб модель розпізнала"],
     "Покладіть предмет на однотонний фон, зніміть при гарному світлі, щоб предмет займав більшу частину кадру. "
     "Уникайте відблисків і сильного розмиття."),
    ("why_sort", ["навіщо сортувати відходи", "чому сортування важливе", "яка користь від сортування",
                  "чим шкодять сміттєзвалища", "що буде якщо не сортувати"],
     "Сортування дає змогу переробити частину відходів замість захоронення. Це зменшує навантаження на полігони, "
     "економить сировину та енергію й зменшує забруднення ґрунту й води."),
    ("bins_colors", ["які кольори контейнерів", "що означають кольори баків", "жовтий синій зелений контейнер",
                     "у який бак що кидати", "кольори сміттєвих баків"],
     "Поширена схема в Україні: **жовтий** — пластик і метал, **синій** — папір і картон, **зелений** — скло. "
     "Конкретні правила залежать від вашої громади, тому перевірте місцеві."),
    ("plastic_caps", ["чи треба знімати кришку з пляшки", "кришечки від пляшок", "що робити з кришками",
                      "корки від пляшок", "кришка пластикова"],
     "Правила різняться. Кришки роблять з іншого пластику, ніж пляшки, тому деякі переробники просять "
     "здавати їх окремо, інші приймають закриту кришку на сплюснутій пляшці. Перевірте правила свого оператора."),
    ("pizza_box", ["чи можна здавати коробку з-під піци", "жирний папір", "брудний картон",
                   "засмальцьована упаковка", "коробка від піци куди"],
     "Чисті частини коробки йдуть у картон, а жирні й забруднені — у змішані відходи, бо жир псує переробку."),
    ("tetrapak", ["тетрапак", "пакети з-під молока і соку", "картонні пакети від соку",
                  "куди тетра пак", "багатошарова упаковка"],
     "Це багатошарова упаковка (картон, пластик, інколи алюміній). У багатьох громадах її збирають окремо "
     "або з папером. Спорожніть і сплюсніть, а точне місце дізнайтесь у місцевого оператора."),
    ("bags", ["поліетиленові пакети", "пакети з супермаркету", "целофан", "що робити з пакетами",
              "пластикові пакети куди"],
     "Чисті й сухі пакети та плівка — це пластик, але приймають їх не скрізь. Найкраще використовувати "
     "пакети багаторазово або брати сумку-шопер."),
    ("broken_glass", ["розбите скло", "розбита склянка", "як викинути битий посуд", "уламки скла",
                      "скляний посуд розбився"],
     "Склотара (пляшки, банки) переробляється, а посуд, дзеркала й віконне скло — ні, тому їх не змішують з тарою. "
     "Уламки загорніть у щільний папір, щоб ніхто не порізався, і викиньте у змішані."),
    ("battery_danger", ["чому батарейки шкідливі", "чим небезпечні батарейки", "що буде якщо викинути батарейку",
                        "шкода від акумуляторів", "чому не можна батарейки в смітник"],
     "Батарейки містять важкі метали (свинець, кадмій, ртуть, залежно від типу), а літієві ще й можуть "
     "спричинити пожежу. На полігоні вони забруднюють ґрунт і воду, тому їх здають окремо."),
    ("bulbs", ["енергозберігаючі лампи", "куди діти старі лампочки", "люмінесцентні лампи", "ртутні лампи",
               "перегоріла лампочка"],
     "Люмінесцентні та енергозберігаючі лампи містять ртуть, тому їх здають у спеціальні пункти. "
     "Старі лампи розжарювання можна у змішані відходи."),
    ("ewaste", ["куди здати старий телефон", "електроніка і техніка", "стара техніка утилізація",
                "зламаний ноутбук куди", "електронні відходи"],
     "Електроніку не викидають у смітник: вона містить небезпечні речовини й цінні метали. "
     "Шукайте пункти прийому електронних відходів, сервісні центри або акції магазинів."),
    ("clothes", ["що робити зі старим одягом", "одяг утилізація", "куди віддати непотрібні речі",
                 "текстиль куди", "старе взуття", "куди дівати стару куртку", "куди дівати старі речі"],
     "Придатний одяг краще віддати: на благодійність, у секонд-хенд, знайомим. "
     "Зношений текстиль приймають у деяких пунктах збору, а нерозірваний чистий можна пустити на ганчірки."),
    ("medicine", ["прострочені ліки", "куди викинути таблетки", "ліки утилізація", "чи можна змивати ліки",
                  "стара аптечка"],
     "Ліки не змивайте в каналізацію. Запитайте в аптеці, чи приймають вони прострочені ліки. "
     "Якщо немає такої можливості, викидайте у змішані відходи в закритій упаковці."),
    ("compost", ["що таке компост", "як зробити компост", "що можна компостувати", "компостування вдома",
                 "можна компостувати м'ясо"],
     "Компостують овочі, фрукти, каву, чай, листя й траву. Не додають м'ясо, рибу, кістки й молочне, "
     "бо вони приваблюють шкідників і неприємно пахнуть."),
    ("decomp_general", ["терміни розкладання відходів", "за який час розкладається сміття",
                        "скільки років розкладаються відходи", "таблиця розкладання", "як довго розкладається сміття"],
     "Орієнтовні терміни розкладання:\n" + _decomp_lines),
    ("reduce", ["як зменшити кількість відходів", "zero waste", "що зробити щоб менше сміття",
                "поради з екологічного життя", "як жити екологічно", "як менше викидати", "як зменшити сміття вдома"],
     "Простий порядок: **відмовтесь** від зайвого, **зменшіть** споживання, **використайте повторно**, "
     "**переробіть** і лише потім викидайте. Беріть багаторазові сумки, пляшки й контейнери."),
    ("reuse", ["різниця між переробкою і повторним використанням", "reduce reuse recycle",
               "що краще переробка чи повторне використання", "що таке recycle"],
     "Повторне використання (reuse) зазвичай кращe за переробку (recycle): воно не потребує енергії на "
     "перетворення матеріалу. Переробка потрібна, коли предмет вже не можна використати знову."),
    ("co2", ["як сортування зменшує викиди", "який вплив на клімат", "скільки co2 економить переробка",
             "чи допомагає переробка клімату", "переробка і викиди"],
     "Виробництво з вторсировини зазвичай потребує менше енергії, ніж із первинної (особливо алюміній), "
     "тому викидів менше. Цифри у вкладці «Калькулятор» орієнтовні."),
    ("ocean", ["пластик в океані", "мікропластик", "чим шкодить пластик природі", "пластик і тварини",
               "забруднення океану"],
     "Пластик розкладається століттями, а розпадаючись, перетворюється на мікропластик. "
     "Він потрапляє у воду та організми тварин, тому важливо, щоб він не опинявся в природі."),
    ("symbols", ["що означають значки на упаковці", "маркування пластику", "цифри в трикутнику",
                 "pet hdpe що це", "коди пластику", "що означає цифра на упаковці", "що означають цифри на пластику"],
     "Цифри в трикутнику вказують тип пластику. PET (1) і HDPE (2) переробляються найлегше, "
     "а типи 6 і 7 значно складніше."),
    ("aerosol", ["аерозольні балончики", "балончик від дезодоранту", "що робити з балончиками",
                 "порожній балончик куди"],
     "Повністю порожні металеві балончики можна здавати як метал. Не пробивайте й не спалюйте їх. "
     "Балончики із залишками здають як небезпечні відходи."),
    ("oil", ["відпрацьована олія", "куди вилити олію", "смажена олія утилізація", "використана олія"],
     "Не зливайте олію в каналізацію. Охолодіть, перелийте в закриту пляшку й віднесіть у пункт прийому "
     "або викиньте зі змішаними відходами."),
    ("what_recycling", ["що таке переробка", "як переробляють відходи", "що таке рециклінг",
                        "що відбувається з відходами після здачі", "як працює переробка скла"],
     "Переробка — це перетворення відходів на нову сировину: матеріал сортують, подрібнюють, очищають і "
     "використовують для нових виробів. Чим чистіші й краще відсортовані відходи, тим легше їх переробити."),
    ("prepare_waste", ["як підготувати пляшку до здачі", "чи потрібно мити тару", "чи треба промивати відходи",
                       "як підготувати відходи до здачі", "чи потрібно відмивати упаковку"],
     "Достатньо спорожнити й ополоснути від залишків їжі, сплюснути (пляшки, коробки) і висушити. "
     "Ідеально чистою тара бути не мусить, але жирна чи гниюча їжа псує всю партію."),
    ("nlp_how", ["як працює чат", "як ти розумієш запитання", "на чому працює діалоговий модуль",
                 "ти використовуєш chatgpt", "це готовий api", "чи ти використовуєш готовий ші", "ти справжній ші чи готовий сервіс"],
     "Чат працює **без зовнішніх API**. Власна база знань з питаннями та відповідями перетворюється на "
     "TF-IDF вектори (символьні n-грами, добре працює з українською морфологією). Запит порівнюється "
     "з базою за косинусною подібністю, а коли впевненість низька, я чесно про це кажу. "
     "Також я використовую результат комп'ютерного зору як контекст."),
]

CLASS_KEYWORDS = {
    "battery": ["батарейк", "акумулят", "батаре"],
    "biological": ["органік", "харчов", "їжа", "їжу", "їжі", "шкірк", "огризк", "біовідход"],
    "cardboard": ["картон", "коробк"],
    "glass": ["скло", "скля", "склотар"],
    "metal": ["метал", "бляшанк", "консерв", "алюмін", "фольг"],
    "paper": ["папір", "паперов", "газет", "журнал"],
    "plastic": ["пластик", "пластмас", "поліетилен"],
    "trash": ["змішан", "звичайне сміття"],
}
ASPECT_STEMS = {
    "where": ["куди", "контейнер", "викин", "викид", "здат", "сортув", "сортуй"],
    "decomp": ["розклад", "розпад", "довго", "скільки"],
    "prepare": ["підготов", "промит", "промив", "помит", "мити", "очист", "очищ", "сплюсн"],
}
CONTEXT_WORDS = {"це", "цей", "ця", "цю", "його", "її", "цього", "цієї", "цим", "цьому", "ним", "нею",
                 "щойно", "розпізнане", "предмет", "річ"}
NLP_THRESHOLD = 0.40
STOPWORDS = {"як", "що", "чи", "про", "мені", "розкажи", "скажи", "будь", "ласка", "можна", "треба",
             "це", "ти", "ви", "а", "і", "й", "та", "в", "у", "на", "з", "із", "до", "є", "ж", "то",
             "зробити", "робити", "хто"}


def norm(text):
    text = text.lower().replace("’", "'").replace("ʼ", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_kb(text):
    """Нормалізація + прибирання стоп-слів (щоб «як», «про» не давали хибних збігів)."""
    n = norm(text)
    kept = [w for w in n.split() if w not in STOPWORDS]
    return " ".join(kept) if kept else n


def md_to_html(text):
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    return text.replace("\n", "<br>")


@st.cache_resource
def build_nlp():
    questions, intents = [], []
    for i, (_, qs, _) in enumerate(KB):
        for q in qs:
            questions.append(norm_kb(q))
            intents.append(i)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
    X = vec.fit_transform(questions)
    return vec, X, np.array(intents)


def nlp_retrieve(text):
    vec, X, intents = build_nlp()
    sims = cosine_similarity(vec.transform([norm_kb(text)]), X)[0]
    order = np.argsort(-sims)
    top_intents = []
    for j in order:
        if intents[j] not in top_intents:
            top_intents.append(int(intents[j]))
        if len(top_intents) == 3:
            break
    return int(intents[order[0]]), float(sims[order[0]]), top_intents


# Окремий тестовий набір: формулювання, яких НЕМАЄ в базі знань
NLP_EVAL_SET = [
    ("як мені відсортувати відходи вдома", "why_sort"),
    ("розкажи чому батарейки шкідливі для ґрунту", "battery_danger"),
    ("чи можна здавати піцу в картон", "pizza_box"),
    ("чим відрізняється рецикл від повторного використання", "reuse"),
    ("де дізнатися про кольори баків", "bins_colors"),
    ("наскільки ти точний", "accuracy"),
    ("чому ти помилився", "limits"),
    ("як сфоткати щоб краще розпізнало", "photo_tips"),
    ("розкажи про тебе", "about"),
    ("добрий вечір", "greeting"),
    ("що робити з енергозберігаючою лампочкою", "bulbs"),
    ("куди дівати старі кофти", "clothes"),
    ("чи можна кидати таблетки в унітаз", "medicine"),
    ("чи шкодить пластик океанам", "ocean"),
    ("як компостувати кухонні залишки", "compost"),
    ("що означає цифра 1 на пляшці", "symbols"),
    ("як утилізувати старий смартфон", "ewaste"),
    ("ти справжній штучний інтелект чи просто api", "nlp_how"),
    ("для чого цей додаток", "about_project"),
    ("коли розкладеться пластик", "decomp_general"),
    ("дякую тобі", "thanks"),
    ("що робити з використаною олією після смаження", "oil"),
]
NLP_OFFTOPIC_SET = ["яка завтра погода", "хто виграв матч", "як приготувати борщ", "курс долара",
                    "розкажи анекдот", "як вивчити python"]


@st.cache_data
def nlp_evaluation():
    """Тест на нових формулюваннях + відхилення запитів не за темою."""
    names = [k[0] for k in KB]
    ok = 0
    for q, true_name in NLP_EVAL_SET:
        idx, score, _ = nlp_retrieve(q)
        ok += int(names[idx] == true_name and score >= NLP_THRESHOLD)
    rej = sum(int(nlp_retrieve(q)[1] < NLP_THRESHOLD) for q in NLP_OFFTOPIC_SET)
    return (ok / len(NLP_EVAL_SET) * 100, len(NLP_EVAL_SET),
            rej / len(NLP_OFFTOPIC_SET) * 100, len(NLP_OFFTOPIC_SET))



def detect_class(q):
    for cls, stems in CLASS_KEYWORDS.items():
        if any(s in q for s in stems):
            return cls
    return None


def detect_aspect(words):
    for aspect, stems in ASPECT_STEMS.items():
        if any(w.startswith(s) for w in words for s in stems):
            return aspect
    return None


def class_answer(cls, aspect, last=None):
    material, bin_, decomp, desc = CLASS_INFO[cls]
    prefix = ""
    if last:
        prefix = f"Судячи з вашого останнього фото (впевненість моделі {last['conf']:.0f}%), це {material}. "
        if last["conf"] < 60:
            prefix += "Але модель була не певна, тож перевірте результат. "
    if aspect == "where":
        return f"{prefix}**{material}** → {bin_}."
    if aspect == "decomp":
        return f"{prefix}**{material}**: орієнтовний час розкладання — {decomp}."
    if aspect == "prepare":
        return f"{prefix}**{material}**: {desc}"
    return (f"{prefix}**{material}**\n• Контейнер: {bin_}\n• Час розкладання: {decomp}\n💡 {desc}")


class EcoVisionSystemCore:

    @staticmethod
    def analyze_image(pil_image):
        img = pil_image.convert("RGB").resize((224, 224))
        x = np.expand_dims(np.array(img, dtype="float32"), axis=0)
        probs = load_model().predict(x, verbose=0)[0]

        idx = int(np.argmax(probs))
        cls = CLASS_NAMES[idx]
        material, bin_, decomp, desc = CLASS_INFO[cls]
        conf = float(probs[idx]) * 100

        if conf < 60:
            desc = ("⚠️ Модель не впевнена, результат може бути хибним. "
                    "Спробуйте інший ракурс або однотонний фон. ") + desc

        top3 = sorted(zip(CLASS_NAMES, probs), key=lambda t: -t[1])[:3]
        return {
            "cls": cls,
            "material": material,
            "confidence": round(conf, 1),
            "bin": bin_,
            "decomp": decomp,
            "desc": desc,
            "top3": [(CLASS_INFO[n][0], float(p) * 100) for n, p in top3],
        }

    @staticmethod
    def process_nlp_smart(query, last=None):
        """Гібрид: контекст із CV + правила для класів + TF-IDF-пошук по базі знань."""
        qn = norm(query)
        words = qn.split()
        if not words:
            return "Напишіть, будь ласка, запитання 🌱"

        mentioned = detect_class(qn)
        aspect = detect_aspect(words)
        has_ctx_word = any(w in CONTEXT_WORDS for w in words)

        cls, from_ctx = None, False
        if mentioned:
            cls = mentioned
        elif has_ctx_word and last:
            cls, from_ctx = last["cls"], True

        # 1) Клас + чітке запитання (куди / скільки / як підготувати)
        if cls and aspect:
            return class_answer(cls, aspect, last if from_ctx else None)

        if aspect and has_ctx_word and not last and not mentioned:
            return ("Я ще не бачив жодного фото 🙂 Завантажте його у вкладці «Зір», "
                    "і я підкажу, що робити з цим предметом.")

        # 2) Пошук по базі знань
        idx, score, top = nlp_retrieve(query)
        if from_ctx and score < 0.65:
            return class_answer(cls, "full", last)
        if score >= NLP_THRESHOLD:
            return f"{KB[idx][2]}\n\n*(впевненість розуміння: {score:.0%})*"

        # 3) Згаданий клас або контекст без уточнення
        if cls:
            return class_answer(cls, "full", last if from_ctx else None)

        if has_ctx_word and not last:
            return ("Я ще не бачив жодного фото 🙂 Завантажте його у вкладці «Зір», "
                    "і я зможу відповідати на питання про цей предмет.")

        # 4) Чесна невпевненість + підказки
        hints = "\n".join(f"• {KB[i][1][0].capitalize()}" for i in top)
        return ("Не впевнений, що правильно зрозумів 🤔 "
                f"*(найближча тема: {score:.0%})*\nМожливо, ви мали на увазі:\n{hints}")


# ==========================================================
# 3.2 СТРЕС-ТЕСТ МІСТА: прогнозна модель сценаріїв
# ==========================================================
CO2_PER_TON = 1.85  # той самий орієнтовний коефіцієнт, що й у калькуляторі


def fmt(n):
    return f"{n:,.0f}".replace(",", " ")


def simulate_city(pop0_k, kg, rec_frac, part, acc, growth, capacity, months):
    """Помісячний прогноз. Все в тоннах на місяць."""
    t = np.arange(months + 1)
    pop = pop0_k * (1 + growth) ** (t / 12)
    waste = pop * kg * 30                       # тис. осіб * кг * 30 днів = т/міс
    demand = waste * rec_frac * part * acc      # скільки вторсировини реально надходить на лінію
    recycled = np.minimum(demand, capacity)     # лінія не переробить більше за свою потужність
    over = np.where(demand > capacity)[0]
    return {
        "t": t, "waste": waste, "demand": demand, "recycled": recycled,
        "landfill_share": (waste - recycled) / waste * 100,
        "overload": int(over[0]) if len(over) else None,
        "co2": float(recycled.sum() * CO2_PER_TON),
    }


def run_scenarios(p):
    base0 = p["pop"] * p["kg"] * 30 * p["rec"] * p["part"] * p["acc"]
    capacity = base0 * (1 + p["reserve"])
    scenarios = [
        ("Базовий сценарій", p["growth"], p["acc"], "#38BDF8"),
        ("Бум населення (темп ×3)", p["growth"] * 3, p["acc"], "#FACC15"),
        ("Падіння точності ШІ (−25 пп)", p["growth"], max(p["acc"] - 0.25, 0.30), "#F87171"),
        ("Бум + доопрацьований ШІ (97%)", p["growth"] * 3, max(p["acc"], 0.97), "#4ADE80"),
    ]
    results = []
    for name, g, a, color in scenarios:
        r = simulate_city(p["pop"], p["kg"], p["rec"], p["part"], a, g, capacity, p["months"])
        r.update(name=name, color=color, need=(r["demand"].max() / base0 - 1) * 100)
        results.append(r)
    return results, capacity, base0


def svg_chart_img(results, capacity, months):
    """Лінійний графік як SVG-картинка (стабільно відображається в темній темі)."""
    W, H, ml, mr, mt, mb = 900, 380, 85, 20, 20, 50
    ymax = max(max(r["recycled"].max() for r in results), capacity) * 1.12

    def X(t):
        return ml + (W - ml - mr) * t / months

    def Y(v):
        return H - mb - (H - mt - mb) * v / ymax

    p = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' font-family='Helvetica, Arial, sans-serif'>"]
    for i in range(5):
        v = ymax * i / 4
        p.append(f"<line x1='{ml}' x2='{W - mr}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' stroke='#94A3B8' stroke-opacity='0.25'/>")
        p.append(f"<text x='{ml - 10}' y='{Y(v) + 5:.1f}' fill='#E2E8F0' font-size='15' text-anchor='end'>{fmt(v)}</text>")
    for t in range(0, months + 1, max(1, months // 6)):
        p.append(f"<text x='{X(t):.1f}' y='{H - mb + 26}' fill='#E2E8F0' font-size='15' text-anchor='middle'>{t}</text>")
    p.append(f"<text x='{(ml + W - mr) / 2:.0f}' y='{H - 8}' fill='#A7F3D0' font-size='15' text-anchor='middle'>місяць прогнозу</text>")
    cy = Y(capacity)
    p.append(f"<line x1='{ml}' x2='{W - mr}' y1='{cy:.1f}' y2='{cy:.1f}' stroke='#FFFFFF' stroke-width='2' stroke-dasharray='8 6'/>")
    p.append(f"<text x='{ml + 8}' y='{cy - 8:.1f}' fill='#FFFFFF' font-size='14'>Потужність лінії</text>")
    for r in results:
        pts = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in zip(r["t"], r["recycled"]))
        p.append(f"<polyline points='{pts}' fill='none' stroke='{r['color']}' stroke-width='3.5' stroke-linejoin='round'/>")
    p.append("</svg>")
    b64 = base64.b64encode("".join(p).encode("utf-8")).decode()
    return (f"<div style='background:rgba(15,23,42,0.85);border-radius:14px;padding:10px;'>"
            f"<img src='data:image/svg+xml;base64,{b64}' style='width:100%;'></div>")


# ==========================================================
# Стан сесії
# ==========================================================
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "agent",
         "text": "👋 **Вітаю! Еко-Агент EcoVision AI готовий до роботи.**\n"
                 "Завантажте фото відходу у вкладці «Зір», і модель визначить його тип. "
                 "Потім можна питати тут: «куди це викинути?»"}
    ]

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "👁️ 1. Зір (CV)"

if "chat_input_val" not in st.session_state:
    st.session_state.chat_input_val = ""


def handle_chat_click():
    user_text = st.session_state.chat_input_val.strip()
    if user_text:
        st.session_state.chat_messages.append({"role": "user", "text": user_text})
        last = None
        if st.session_state.get("last_cls"):
            last = {"cls": st.session_state.last_cls, "conf": st.session_state.get("last_conf", 0.0)}
        response_text = EcoVisionSystemCore.process_nlp_smart(user_text, last)
        st.session_state.chat_messages.append({"role": "agent", "text": response_text})
        st.session_state.chat_input_val = ""


# ==========================================================
# 4. ГОЛОВНИЙ ЕКРАН
# ==========================================================
header_col1, header_col2 = st.columns([0.08, 0.92], gap="small")
with header_col1:
    st.image("https://img.icons8.com/color/96/environment-care.png", width=65)
with header_col2:
    st.markdown('<p class="main-title" style="font-size: 38px; margin-top: 5px;">EcoVision AI — Autonomous Environmental Agent</p>', unsafe_allow_html=True)

st.markdown('<p class="subtitle">Система комп’ютерного зору, NLP-аналізу та прогностики для GENIUS Olympiad</p>', unsafe_allow_html=True)
st.markdown("---")

tab_names = [
    "👁️ 1. Зір (CV)",
    "💬 2. Чат (NLP)",
    "📈 3. Калькулятор",
    "🧪 4. Стрес-тест",
    "🛠 5. Архітектура",
]

cols = st.columns(len(tab_names))
for i, name in enumerate(tab_names):
    with cols[i]:
        if st.button(name, key=f"tab_btn_{i}"):
            st.session_state.active_tab = name

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="animated-section">', unsafe_allow_html=True)

# ----------------------------------------------------------
# РОЗДІЛ 1: КОМП'ЮТЕРНИЙ ЗІР
# ----------------------------------------------------------
if st.session_state.active_tab == "👁️ 1. Зір (CV)":
    st.subheader("Модуль комп’ютерного зору (MobileNetV2, донавчена на 8 класах відходів)")
    st.write("Завантажте фото відходу або зробіть знімок через веб-камеру:")

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("### 📥 Вхідний потік даних")
        src_mode = st.radio("Джерело зображення:", ["📁 Завантажити файл", "📷 Веб-камера"], key="cv_src")
        uploaded_img = None

        if "Завантажити" in src_mode:
            uploaded_img = st.file_uploader("Оберіть файл (JPG, PNG)", type=["jpg", "jpeg", "png"], key="cv_file")
        else:
            uploaded_img = st.camera_input("Зробіть знімок предмета", key="cv_cam")

        st.markdown("<br>", unsafe_allow_html=True)
        run_cv_btn = st.button("🚀 Запустити аналіз", type="primary", key="cv_btn")

    with col_r:
        st.subheader("🔍 Результат класифікації")
        if uploaded_img is not None:
            image_obj = Image.open(uploaded_img)
            st.image(image_obj, caption="📸 Кадр завантажено", use_container_width=True)

            if run_cv_btn:
                if not os.path.exists(MODEL_PATH):
                    st.error("Файл ecovision_model.keras не знайдено поруч з app.py.")
                else:
                    with st.spinner("Нейромережа аналізує зображення..."):
                        cv_res = EcoVisionSystemCore.analyze_image(image_obj)

                    # Зберігаємо результат, щоб чат знав, що побачила модель
                    st.session_state.last_cls = cv_res["cls"]
                    st.session_state.last_conf = cv_res["confidence"]

                    m1, m2 = st.columns(2)
                    m1.metric("Впевненість моделі", f"{cv_res['confidence']}%")
                    m2.metric("Орієнтовний час розпаду", cv_res["decomp"])

                    top_lines = "<br>".join(f"&nbsp;&nbsp;{n}: {p:.1f}%" for n, p in cv_res["top3"])
                    st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(74, 222, 128, 0.4); padding: 30px; border-radius: 18px; margin-top: 15px;">
                    🤖 <b>Висновок моделі:</b><br><br>
                    • <b>Клас:</b> {cv_res['material']}<br>
                    • <b>Контейнер:</b> <span style="color: #4ADE80; font-weight: bold;">{cv_res['bin']}</span><br><br>
                    💡 {cv_res['desc']}<br><br>
                    <b>Топ-3 ймовірності:</b><br>{top_lines}
                    </div>
                    """, unsafe_allow_html=True)
                    st.success("✅ Результат передано в чат. Можете спитати: «куди це викинути?»")
        else:
            st.info("👆 Завантажте фото або зробіть знімок ліворуч і натисніть кнопку запуску.")

# ----------------------------------------------------------
# РОЗДІЛ 2: ЧАТ
# ----------------------------------------------------------
elif st.session_state.active_tab == "💬 2. Чат (NLP)":
    st.subheader("Діалоговий модуль (власна NLP-модель, без зовнішніх API)")
    st.write("Спілкуйтеся з Еко-Агентом:")

    if st.session_state.get("last_cls"):
        _m = CLASS_INFO[st.session_state.last_cls][0]
        st.info(f"🔗 Контекст із CV: {_m} ({st.session_state.get('last_conf', 0):.0f}%). "
                "Можна питати «куди це викинути?», «скільки це розкладається?»")

    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user"><b>👤 Ви:</b><br>{html.escape(msg["text"])}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-agent"><b>🤖 EcoVision AI:</b><br>{md_to_html(msg["text"])}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<b>Введіть ваше запитання до агента:</b>", unsafe_allow_html=True)

    c_input, c_btn = st.columns([4, 1], gap="medium")
    with c_input:
        st.text_input("Ваше запитання", placeholder="Наприклад: що робити з тетрапаком?",
                      key="chat_input_val", label_visibility="collapsed")
    with c_btn:
        st.button("Надіслати 🚀", on_click=handle_chat_click, key="chat_send_btn")

    st.markdown("---")
    st.markdown("### 📊 Наскільки точний NLP-модуль")
    _a, _na, _r, _nr = nlp_evaluation()
    e1, e2, e3 = st.columns(3)
    e1.metric("Тем у базі знань", len(KB))
    e2.metric(f"Тест на нових запитах ({_na})", f"{_a:.0f}%")
    e3.metric(f"Відхилено чужі запити ({_nr})", f"{_r:.0f}%")
    st.markdown("""
    <div class="info-card-bright">
    Тестовий набір складається з формулювань, яких <b>немає в базі знань</b>. Запит зараховується,
    якщо модель знайшла правильну тему з упевненістю вище порога. Чужі теми (погода, спорт) мають відхилятися.
    Набір невеликий і складений автором, тому на реальних запитах точність може бути нижчою.
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------
# РОЗДІЛ 3: КАЛЬКУЛЯТОР
# ----------------------------------------------------------
elif st.session_state.active_tab == "📈 3. Калькулятор":
    st.subheader("📈 Калькулятор орієнтовного ефекту від сортування відходів")
    st.markdown("""
    <div class="info-card-bright">
    💡 Це модельна <b>оцінка</b>, а не вимірювання. Коефіцієнти нижче орієнтовні
    і будуть замінені на значення з джерел (наприклад, EPA WARM).
    </div>
    """, unsafe_allow_html=True)

    col_c1, col_c2 = st.columns(2, gap="large")
    with col_c1:
        st.markdown("### ⚙️ Вхідні параметри")
        households = st.slider("Кількість домогосподарств:", 1000, 200000, 25000, key="c_hh")
        daily_waste = st.slider("Відходів від однієї сім'ї на день (кг):", 1.0, 10.0, 3.5, key="c_w")
        ai_accuracy = st.slider("Частка правильно відсортованого (%):", 70, 99, 92, key="c_acc")

    with col_c2:
        st.markdown("### 📊 Орієнтовний ефект за місяць:")
        recycled_tons = (households * daily_waste * 30 * (ai_accuracy / 100.0)) / 1000.0
        co2_saved = recycled_tons * 1.85
        energy_saved = recycled_tons * 2.95
        landfill_saved = recycled_tons * 1.40

        st.metric("Відсортована вторсировина", f"{recycled_tons:,.1f} т/міс")
        st.metric("Зменшення викидів (орієнтовно)", f"{co2_saved:,.1f} т CO2")
        st.metric("Збережена енергія (орієнтовно)", f"{energy_saved:,.1f} МВт·год")
        st.metric("Об'єм полігонів (орієнтовно)", f"{landfill_saved:,.1f} м³")

# ----------------------------------------------------------
# РОЗДІЛ 4: СТРЕС-ТЕСТ
# ----------------------------------------------------------
elif st.session_state.active_tab == "🧪 4. Стрес-тест":
    st.subheader("🧪 Стрес-тест міста: коли система сортування «ламається»")
    st.markdown("""
    <div class="info-card-bright">
    Прогнозна модель показує, що буде з сортуванням у місті за різних сценаріїв: швидке зростання населення
    або падіння точності ШІ-підказки. Головне питання: <b>коли сортувальна лінія перевантажується</b>
    і скільки відходів потрапляє на полігон. Точність ШІ за замовчуванням узята з тесту нашої моделі (≈92%),
    інші параметри орієнтовні, їх можна змінювати.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("### 🏙️ Місто")
        s_pop = st.slider("Населення, тис. осіб:", 50, 1000, 300, 10, key="s_pop")
        s_growth = st.slider("Зростання населення, % на рік:", 0.0, 8.0, 3.0, 0.5, key="s_gr")
        s_kg = st.slider("Відходів на людину, кг на день:", 0.5, 2.0, 1.1, 0.1, key="s_kg")
        s_months = st.slider("Горизонт прогнозу, місяців:", 12, 60, 36, 6, key="s_m")
    with c2:
        st.markdown("### ⚙️ Система сортування")
        s_rec = st.slider("Частка відходів, придатних до переробки, %:", 10, 60, 35, key="s_rec")
        s_part = st.slider("Частка жителів, що сортують, %:", 10, 100, 40, key="s_part")
        s_acc = st.slider("Точність ШІ-підказки, %:", 60, 99, 92, key="s_acc")
        s_res = st.slider("Запас потужності лінії, %:", 0, 100, 25, key="s_res")

    params = {"pop": s_pop, "growth": s_growth / 100, "kg": s_kg, "months": s_months,
              "rec": s_rec / 100, "part": s_part / 100, "acc": s_acc / 100, "reserve": s_res / 100}
    results, capacity, base0 = run_scenarios(params)

    st.markdown("### 📈 Скільки вторсировини переробляється щомісяця, т")
    st.markdown(svg_chart_img(results, capacity, s_months), unsafe_allow_html=True)
    legend = "".join(
        f"<span style='margin-right:22px;white-space:nowrap;'>"
        f"<span style='display:inline-block;width:14px;height:14px;border-radius:3px;background:{r['color']};"
        f"margin-right:8px;vertical-align:middle;'></span>{r['name']}</span>" for r in results)
    st.markdown(f"<div style='margin:10px 0 20px 0;'>{legend}</div>", unsafe_allow_html=True)

    st.markdown(f"### 📋 Підсумок через {s_months} міс.")
    base_end = results[0]["landfill_share"][-1]
    rows = ""
    for r in results:
        end_share = r["landfill_share"][-1]
        share_bg = ("rgba(22,163,74,0.85)" if end_share <= base_end + 2 else
                    "rgba(202,138,4,0.9)" if end_share <= base_end + 10 else "rgba(185,28,28,0.9)")
        if r["overload"] is None:
            ov_txt, ov_bg = "Витримує", "rgba(22,163,74,0.85)"
        else:
            ov_txt, ov_bg = f"Місяць {r['overload']}", "rgba(185,28,28,0.9)"
        neutral = "rgba(15,23,42,0.85)"
        rows += (f"<tr><td class='kind'><span style='color:{r['color']};'>●</span> {r['name']}</td>"
                 f"<td class='cell' style='background:{neutral}'>{fmt(r['recycled'][-1])}</td>"
                 f"<td class='cell' style='background:{share_bg}'>{end_share:.0f}%</td>"
                 f"<td class='cell' style='background:{ov_bg}'>{ov_txt}</td>"
                 f"<td class='cell' style='background:{neutral}'>{fmt(r['co2'])}</td></tr>")
    st.markdown(
        "<div class='heat-wrap'><table class='heat-table'><tr><th></th>"
        "<th>Переробка в кінці, т/міс</th><th>Відходів на полігон</th>"
        "<th>Перевантаження лінії</th><th>CO₂ за період, т (орієнт.)</th></tr>"
        f"{rows}</table></div>", unsafe_allow_html=True)

    lines = []
    for r in results:
        if r["overload"] is not None:
            lines.append(f"• <b>{r['name']}</b>: лінія перевантажується на {r['overload']}-му місяці. "
                         f"Щоб витримати, потрібен запас потужності близько {r['need']:.0f}% "
                         f"(зараз {s_res}%).")
    if not lines:
        lines.append("• Жоден сценарій не перевантажує лінію за обраний горизонт.")
    per_pp = s_pop * s_kg * 30 * params["rec"] * params["part"] * 0.01
    lines.append(f"• Кожен 1 пп точності ШІ дає приблизно {per_pp:,.0f} т вторсировини на місяць на старті.".replace(",", " "))
    lines.append("• Висновок: точність ШІ допомагає лише тоді, коли лінія встигає переробляти потік, "
                 "тому зростання потужності й точність потрібно планувати разом.")
    st.markdown("<div class='info-card-bright'>📌 <b>Що показує стрес-тест:</b><br>" + "<br>".join(lines) +
                "<br><br><i>Це модельна оцінка за орієнтовними припущеннями, а не вимірювання.</i></div>",
                unsafe_allow_html=True)

# ----------------------------------------------------------
# РОЗДІЛ 5: АРХІТЕКТУРА
# ----------------------------------------------------------
elif st.session_state.active_tab == "🛠 5. Архітектура":
    st.subheader("🛠 Архітектура системи та внесок розробника")
    st.markdown("""
    ### 🎯 1. Яку проблему вирішує проєкт
    Неправильне сортування побутових відходів збільшує навантаження на полігони.
    Система визначає тип відходу за фото й підказує, до якого контейнера його віднести.

    ---
    ### 🧠 2. Комп'ютерний зір
    * **Архітектура:** MobileNetV2, попередньо навчена на ImageNet (transfer learning).
    * **Дані:** публічний датасет Garbage Classification (Kaggle), 8 класів, 8213 фото.
    * **Поділ даних:** 70% навчання / 15% валідація / 15% тест.
    * **Навчання:** аугментація, ваги класів, дворівневе навчання (нова голова, потім fine-tuning).
    * **Результат на тесті:** близько 92% accuracy (1232 фото, які модель не бачила).

    ---
    ### 💬 3. NLP-модуль (власна розробка, без зовнішніх API)
    * **База знань:** власноруч складені питання й відповіді про сортування та екологію.
    * **Модель:** TF-IDF із символьними n-грамами (2–5) і пошук найближчого запитання за косинусною подібністю.
    * **Інтеграція з CV:** чат знає, що модель щойно розпізнала, і розуміє запитання на кшталт «куди це викинути?».
    * **Валідація:** окремий тестовий набір нових формулювань і перевірка відхилення чужих тем.
    * **Надійність:** поріг впевненості; нижче нього агент чесно каже, що не впевнений, і пропонує схожі теми.

    ---
    ### 🧪 4. Валідація
    * **Стрес-тест системи:** помісячний прогноз для 4 сценаріїв (зростання населення, падіння точності ШІ, доопрацьований ШІ), пошук місяця, коли сортувальна лінія перевантажується.
    * Точність ШІ береться з тесту моделі (≈92%), решта параметрів орієнтовні й змінюються повзунками.

    ---
    ### ⚠️ 5. Обмеження
    * Датасет складається переважно з фото на однотонному фоні, тому на реальних знімках точність нижча.
    * Модель плутає пластик і скло (прозорі поверхні), а також блискучі предмети та метал.
    * Нетипові форми, розбиті або зім'яті предмети розпізнаються гірше.
    * NLP-модуль знає лише теми зі своєї бази знань. Тестовий набір невеликий і складений автором.
    """)
    _a, _na, _r, _nr = nlp_evaluation()
    st.markdown(f"**Валідація NLP:** {_a:.0f}% правильних тем на {_na} нових формулюваннях; "
                f"{_r:.0f}% запитів не за темою коректно відхилено ({_nr} шт.).")

st.markdown("</div>", unsafe_allow_html=True)
