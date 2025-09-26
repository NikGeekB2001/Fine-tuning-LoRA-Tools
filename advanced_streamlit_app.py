import sqlite3
import os
from datetime import datetime, timedelta
from advanced_medical_assistant import AdvancedMedicalAssistant
import warnings
warnings.filterwarnings("ignore")

# Настройка страницы
st.set_page_config(
    page_title="Продвинутый медицинский помощник",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стилизация
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .appointment-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .medical-response {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        white-space: pre-wrap;
    }
    .loading-spinner {
        color: #007bff;
        font-weight: bold;
    }
    .model-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .status-ready {
        background-color: #d4edda;
        color: #155724;
    }
    .status-loading {
        background-color: #fff3cd;
        color: #856404;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
    }
    .entity-tag {
        display: inline-block;
        background-color: #e9ecef;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedMedicalApp:
    def __init__(self):
        self.db_path = "appointments.db"
        self.assistant = None
        self.initialize_database()
        self.load_assistant()
    
    def initialize_database(self):
        """Инициализация базы данных для записей к врачу"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Проверяем существующую структуру таблицы
        cursor.execute("PRAGMA table_info(appointments)")
        existing_columns = [row[1] for row in cursor.fetchall()]

        if not existing_columns:
            # Создаем таблицу если она не существует
            cursor.execute('''
                CREATE TABLE appointments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_name TEXT NOT NULL,
                    doctor_type TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    symptoms TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        
        conn.commit()
        conn.close()
    
    def load_assistant(self):
        """Загрузка медицинского ассистента"""
        with st.spinner("Загрузка медицинских моделей..."):
            try:
                self.assistant = AdvancedMedicalAssistant()
                if self.assistant.models_loaded:
                    st.success("Медицинские модели успешно загружены!")
                else:
                    st.error("Не удалось загрузить медицинские модели")
            except Exception as e:
                st.error(f"Ошибка при загрузке ассистента: {e}")
    
    def book_appointment(self, patient_name: str, doctor_type: str, date: str, time: str, symptoms: str = "") -> dict:
        """Запись на прием"""
        try:
            # Валидация даты и времени
            datetime.strptime(date, "%Y-%m-%d")
            datetime.strptime(time, "%H:%M")

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO appointments (patient_name, doctor_type, date, time, symptoms)
                VALUES (?, ?, ?, ?, ?)
            ''', (patient_name, doctor_type, date, time, symptoms))

            appointment_id = cursor.lastrowid
            conn.commit()
            conn.close()

            return {
                "success": True,
                "appointment_id": appointment_id,
                "message": f"Запись успешно создана! Номер записи: {appointment_id} Пациент: {patient_name} Врач: {doctor_type} Дата: {date} Время: {time}"
            }

        except ValueError:
            return {
                "success": False,
                "message": "Некорректный формат даты или времени. Используйте YYYY-MM-DD и HH:MM."
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Ошибка при создании записи: {str(e)}"
            }
    
    def get_appointments(self, patient_name: str = None) -> list:
        """Получение списка записей"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if patient_name:
            cursor.execute('''
                SELECT * FROM appointments
                WHERE patient_name = ?
                ORDER BY date DESC, time DESC
            ''', (patient_name,))
        else:
            cursor.execute('''
                SELECT * FROM appointments
                ORDER BY date DESC, time DESC
            ''')

        appointments = cursor.fetchall()
        conn.close()

        # Получаем информацию о колонках
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(appointments)")
        columns = cursor.fetchall()
        conn.close()

        column_names = [col[1] for col in columns]

        result = []
        for apt in appointments:
            apt_dict = {}
            for i, col_name in enumerate(column_names):
                if i < len(apt):
                    apt_dict[col_name] = apt[i]
                else:
                    apt_dict[col_name] = None
            result.append(apt_dict)

        return result
    
    def cancel_appointment(self, appointment_id: int) -> dict:
        """Отмена записи"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('DELETE FROM appointments WHERE id = ?', (appointment_id,))
            conn.commit()

            if cursor.rowcount > 0:
                conn.close()
                return {
                    "success": True,
                    "message": f"Запись #{appointment_id} успешно отменена."
                }
            else:
                conn.close()
                return {
                    "success": False,
                    "message": f"Запись #{appointment_id} не найдена."
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Ошибка при отмене записи: {str(e)}"
            }

def main():
    st.markdown('<div class="main-header">Продвинутый медицинский помощник</div>', unsafe_allow_html=True)

    # Инициализация приложения
    if 'app' not in st.session_state:
        st.session_state.app = AdvancedMedicalApp()
    
    app = st.session_state.app
    
    # Отображение статуса моделей
    if app.assistant and app.assistant.models_loaded:
        st.markdown('<div class="model-status status-ready">Медицинские модели готовы к работе</div>', unsafe_allow_html=True)
    elif app.assistant:
        st.markdown('<div class="model-status status-loading">Модели загружаются...</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="model-status status-error">Модели не загружены</div>', unsafe_allow_html=True)

    # Боковая панель
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Управление записями</div>', unsafe_allow_html=True)

        # Создание новой записи
        st.subheader("Запись на прием")

        with st.form("appointment_form"):
            patient_name = st.text_input("Имя пациента", placeholder="Введите ваше имя")
            doctor_type = st.selectbox(
                "Тип врача",
                ["Терапевт", "Кардиолог", "Невролог", "Дерматолог", "Офтальмолог", "Другое"]
            )

            col1, col2 = st.columns(2)
            with col1:
                date = st.date_input("Дата", min_value=datetime.now().date())
            with col2:
                time = st.time_input("Время")

            symptoms = st.text_area("Опишите ваши симптомы", placeholder="Например: болит голова, температура 38.5")

            submitted = st.form_submit_button("Записаться на прием")

            if submitted:
                if not patient_name.strip():
                    st.error("Пожалуйста, введите имя пациента")
                else:
                    result = app.book_appointment(
                        patient_name.strip(),
                        doctor_type,
                        date.strftime("%Y-%m-%d"),
                        time.strftime("%H:%M"),
                        symptoms.strip()
                    )

                    if result["success"]:
                        st.success(result["message"])
                        st.rerun()
                    else:
                        st.error(result["message"])

        # Поиск записей
        st.subheader("Поиск записей")
        search_name = st.text_input("Поиск по имени пациента", placeholder="Введите имя для поиска")

        if st.button("Найти записи"):
            if search_name.strip():
                appointments = app.get_appointments(search_name.strip())
                if appointments:
                    st.success(f"Найдено {len(appointments)} записей")
                    for apt in appointments:
                        st.markdown(f'''
                        <div class="appointment-card">
                            <strong>Запись #{apt["id"]}</strong><br>
                            Пациент: {apt["patient_name"]}<br>
                            Врач: {apt["doctor_type"]}<br>
                            Дата: {apt["date"]}<br>
                            Время: {apt["time"]}<br>
                            Симптомы: {apt["symptoms"] or "Не указаны"}
                        </div>
                        ''', unsafe_allow_html=True)
                else:
                    st.info("Записи не найдены")
            else:
                st.warning("Введите имя для поиска")

    # Основная область
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Чат с медицинским помощником")

        # Инициализация истории чата
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Отображение истории чата
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**Вы:** {message['content']}")
            else:
                st.markdown(f'<div class="medical-response">**Медицинский помощник:** {message["content"]}</div>', unsafe_allow_html=True)

        # Ввод нового сообщения
        user_input = st.text_area("Задайте ваш вопрос:", height=100, 
                                 placeholder="Например: Ногу ушиб какая мазь нужна? Что делать при температуре 38.5?")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Отправить", type="primary"):
                if user_input.strip() and app.assistant and app.assistant.models_loaded:
                    # Добавляем сообщение пользователя
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_input
                    })

                    # Обрабатываем вопрос с помощью продвинутого ассистента
                    with st.spinner("Анализ вопроса..."):
                        try:
                            response = app.assistant.process_question(user_input)
                        except Exception as e:
                            response = f"Произошла ошибка при обработке вопроса: {str(e)}"

                    # Добавляем ответ ассистента
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })

                    st.rerun()
                elif not app.assistant or not app.assistant.models_loaded:
                    st.error("Медицинские модели не загружены. Пожалуйста, подождите или перезапустите приложение.")
                else:
                    st.warning("Пожалуйста, введите сообщение")

        with col_b:
            if st.button("Очистить чат"):
                st.session_state.chat_history = []
                st.rerun()

    with col2:
        st.subheader("Статистика")

        # Получаем статистику
        all_appointments = app.get_appointments()
        today_appointments = [apt for apt in all_appointments if apt["date"] == datetime.now().strftime("%Y-%m-%d")]

        col_stats1, col_stats2, col_stats3 = st.columns(3)

        with col_stats1:
            st.metric("Всего записей", len(all_appointments))

        with col_stats2:
            st.metric("Записей на сегодня", len(today_appointments))

        with col_stats3:
            st.metric("Типов врачей", len(set([apt["doctor_type"] for apt in all_appointments])))

        # Информация о моделях
        st.subheader("Информация о моделях")
        
        if app.assistant and app.assistant.models_loaded:
            st.success("Все модели загружены")
            st.markdown("""
            **Активные модели:**
            - RuGPT-3 Large: Генерация ответов
            - RuBERT Large SQuAD 2: Вопрос-ответ и NER
            - FAISS: Семантический поиск
            - LoRA: Адаптация NER под медицинские данные
            """)
        else:
            st.error("Модели не загружены")

        # Последние записи
        st.subheader("Последние записи")
        recent_appointments = all_appointments[:5]

        if recent_appointments:
            for apt in recent_appointments:
                with st.expander(f"Запись #{apt['id']} - {apt['patient_name']}"):
                    st.write(f"**Врач:** {apt['doctor_type']}")
                    st.write(f"**Дата:** {apt['date']}")
                    st.write(f"**Время:** {apt['time']}")
                    st.write(f"**Симптомы:** {apt['symptoms'] or 'Не указаны'}")

                    if st.button(f"Отменить запись #{apt['id']}", key=f"cancel_{apt['id']}"):
                        result = app.cancel_appointment(apt['id'])
                        if result["success"]:
                            st.success(result["message"])
                            st.rerun()
                        else:
                            st.error(result["message"])
        else:
            st.info("Записи не найдены")

if __name__ == "__main__":
    main()