# advanced_medical_assistant.py
import os
import torch
import warnings
import re
warnings.filterwarnings("ignore")

# Импортируем инструменты из medical_tools.py
try:
    from medical_tools import (
        find_info_in_knowledge_base,
        extract_precise_answer,
        analyze_medical_entities
    )
    print("OK Инструменты из medical_tools.py загружены")
except ImportError as e:
    print(f"ERROR Не удалось загрузить инструменты: {e}")
    find_info_in_knowledge_base = None
    extract_precise_answer = None
    analyze_medical_entities = None

class AdvancedMedicalAssistant:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Используется устройство: {self.device}")
        
        self.models_loaded = False
        self.load_models()
        
    def load_models(self):
        """Загрузка всех необходимых моделей"""
        try:
            print("Загрузка моделей...")
            
            # Импортируем transformers только когда нужно
            from transformers import (
                AutoTokenizer, 
                AutoModelForCausalLM, 
                AutoModelForQuestionAnswering,
                AutoModelForTokenClassification,
                pipeline
            )
            from peft import PeftModel
            
            # 1. Генеративная модель для диалогов
            print("1. Загрузка генеративной модели...")
            self.gpt_model_name = "sberbank-ai/rugpt3large_based_on_gpt2"
            self.gpt_tokenizer = AutoTokenizer.from_pretrained(self.gpt_model_name)
            if self.gpt_tokenizer.pad_token is None:
                self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
            self.gpt_model = AutoModelForCausalLM.from_pretrained(
                self.gpt_model_name,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # 2. QA модель для извлечения точных ответов
            print("2. Загрузка QA модели...")
            self.qa_model_name = "Den4ikAI/rubert_large_squad_2"
            self.qa_tokenizer = AutoTokenizer.from_pretrained(self.qa_model_name)
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained(
                self.qa_model_name,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=False
            ).to(self.device)
            
            # 3. NER модель с LoRA для распознавания сущностей
            print("3. Загрузка NER модели...")
            self.ner_model_name = "Den4ikAI/rubert_large_squad_2"
            self.ner_tokenizer = AutoTokenizer.from_pretrained(self.ner_model_name)
            self.ner_base_model = AutoModelForTokenClassification.from_pretrained(
                self.ner_model_name,
                num_labels=13,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=False
            ).to(self.device)
            
            # Попытка загрузить LoRA адаптер
            try:
                self.ner_model = PeftModel.from_pretrained(self.ner_base_model, "./models/lora_adapter")
                print("   OK LoRA адаптер для NER загружен")
            except:
                print("   WARNING LoRA адаптер не найден, используем базовую модель")
                self.ner_model = self.ner_base_model
            
            # 4. Создание pipeline'ов
            print("4. Создание pipeline'ов...")
            self.text_generator = pipeline(
                "text-generation",
                model=self.gpt_model,
                tokenizer=self.gpt_tokenizer,
                max_length=1024,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.gpt_tokenizer.eos_token_id
            )
            
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.qa_model,
                tokenizer=self.qa_tokenizer
            )
            
            self.models_loaded = True
            print("OK Все модели успешно загружены!")
            
        except Exception as e:
            print(f"ERROR Ошибка при загрузке моделей: {e}")
            self.models_loaded = False
    
    def analyze_entities(self, text: str) -> str:
        """Анализ медицинских сущностей"""
        if not self.models_loaded or analyze_medical_entities is None:
            return "Модель NER не доступна"
        
        try:
            return analyze_medical_entities.invoke(text)
        except Exception as e:
            return f"Ошибка при анализе сущностей: {e}"
    
    def search_knowledge(self, query: str) -> str:
        """Поиск в базе знаний"""
        if not self.models_loaded or find_info_in_knowledge_base is None:
            return "Поиск в базе знаний не доступен"
        
        try:
            return find_info_in_knowledge_base.invoke(query)
        except Exception as e:
            return f"Ошибка при поиске информации: {e}"
    
    def extract_answer(self, context: str, question: str) -> str:
        """Извлечение ответа из контекста"""
        if not self.models_loaded:
            return "Модели не загружены"
            
        try:
            # Используем встроенный QA pipeline
            result = self.qa_pipeline(question=question, context=context)
            return result['answer']
        except Exception as e:
            return f"Ошибка при извлечении ответа: {e}"
    
    def get_question_type(self, question: str) -> str:
        """Определение типа вопроса"""
        question_lower = question.lower()

        # Проверка на приветствия
        if any(greeting in question_lower for greeting in ["привет", "здравствуйте", "добрый день", "доброе утро", "добрый вечер", "хай", "hello", "hi"]):
            return "приветствие"

        if any(keyword in question_lower for keyword in ["мазь", "крем", "гель", "мази", "крема"]):
            return "мазь"
        elif any(keyword in question_lower for keyword in ["таблетка", "препарат", "лекарство", "пилюля", "капсула"]):
            return "таблетка"
        elif any(keyword in question_lower for keyword in ["температура", "жар", "лихорадка", "жарит"]):
            return "температура"
        elif any(keyword in question_lower for keyword in ["боль", "болит", "болеть", "обезболивающее"]):
            return "боль"
        elif any(keyword in question_lower for keyword in ["симптом", "признак", "ознаки", "проявление"]):
            return "симптом"
        elif any(keyword in question_lower for keyword in ["что такое", "что это", "объясните", "расскажите"]):
            return "объяснение"
        elif any(keyword in question_lower for keyword in ["обрезал", "порезал", "рана", "кровотечение"]):
            return "рана"
        else:
            return "общий"

    def generate_generative_response(self, question: str, context: str, entities: str) -> str:
        """Генерация ответа с помощью генеративной модели"""
        if not self.models_loaded:
            return "Генеративная модель не загружена"

        try:
            # Определяем тип вопроса для выбора шаблона
            question_type = self.get_question_type(question)

            # Если генерация не удалась, используем улучшенные шаблоны
            if question_type in ["температура", "мазь", "таблетка", "боль", "рана", "приветствие"]:
                print(f"Используем улучшенный шаблон для типа: {question_type}")
                return self.generate_template_response(question, context, entities, question_type)

            # Для сложных вопросов пытаемся генерировать
            prompt = f"""Ты медицинский ассистент. Дай конкретные, безопасные рекомендации. Всегда подчеркивай консультацию врача.

Вопрос: {question}
Контекст: {context[:200] if context else 'Нет.'}
Сущности: {entities if entities else 'Нет.'}

Ответь в формате:
### Рекомендации
- Конкретные шаги (3-4 пункта)

### Когда к врачу
- Когда срочно (2-3 пункта)

### Важно
Обратитесь к врачу."""

            generated = self.text_generator(
                prompt,
                max_new_tokens=150,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.gpt_tokenizer.eos_token_id,
                repetition_penalty=1.2
            )

            generated_text = generated[0]['generated_text']
            new_text = generated_text[len(prompt):].strip()

            # Мягкий фильтр
            non_medical = ['блог', 'СМИ', 'стихи', 'осень', 'фильм', 'история']
            if any(word in new_text.lower() for word in non_medical):
                print("Генерация содержит не-медицинский контент, используем шаблон")
                return self.generate_template_response(question, context, entities, question_type)

            if len(new_text) < 50 or not "### Рекомендации" in new_text:
                return self.generate_template_response(question, context, entities, question_type)

            response = f"### Медицинская консультация\n\n**Вопрос:** {question}\n\n{new_text}\n\n**Важно:** Эта информация ознакомительная. Обратитесь к врачу."
            return response

        except Exception as e:
            print(f"Ошибка при генерации ответа: {e}")
            return self.generate_template_response(question, context, entities, self.get_question_type(question))

    def generate_template_response(self, question: str, context: str, entities: str, question_type: str) -> str:
        """Генерация ответа на основе улучшенных шаблонов"""
        try:
            response_parts = []

            # Заголовок
            response_parts.append("### Медицинская консультация")
            response_parts.append(f"\n**Вопрос:** {question}")

            # Анализ сущностей
            if entities and "Найденные медицинские сущности:" in entities:
                response_parts.append(f"\n**Анализ сущностей:** {entities}")

            # Получаем улучшенные шаблоны
            templates = self.get_enhanced_templates(question_type)

            # Добавляем конкретные рекомендации
            response_parts.append("\n**Рекомендации:**")
            for rec in templates["рекомендации"]:
                response_parts.append(rec)

            # Точный ответ из контекста
            if context and len(context) > 50:
                try:
                    precise_answer = self.extract_answer(context, question)
                    if precise_answer and len(precise_answer) > 5 and not (question_type == "мазь" and "парацетамол" in precise_answer.lower()):
                        response_parts.append(f"\n**Точный ответ из базы знаний:** {precise_answer}")
                except Exception as e:
                    print(f"Не удалось извлечь точный ответ: {e}")

            # Ключевые предупреждения
            response_parts.append("\n**Предупреждения и когда обращаться к врачу:**")
            for warn in templates["предупреждения"]:
                response_parts.append(warn)

            # Общие рекомендации
            response_parts.append("\n**Общие рекомендации:**")
            response_parts.append("• При устойчивых симптомах обратитесь к врачу")
            response_parts.append("• Соблюдайте режим и пейте больше жидкости")
            response_parts.append("• Не занимайтесь самолечением серьезными препаратами")

            # Важно
            response_parts.append("\n**Важно:**")
            response_parts.append("Эта информация носит ознакомительный характер и не заменяет консультацию врача.")
            response_parts.append("При серьезных симптомах немедленно обратитесь за медицинской помощью.")

            return "\n".join(response_parts)

        except Exception as e:
            return self.get_fallback_response(question)

    def get_enhanced_templates(self, question_type: str) -> dict:
        """Улучшенные шаблоны с более конкретными рекомендациями"""
        templates = {
            "приветствие": {
                "рекомендации": [
                    "Здравствуйте! Я ваш медицинский ИИ-ассистент. Я могу помочь с общими медицинскими вопросами, объяснить симптомы, дать рекомендации по уходу.",
                    "Задайте мне вопрос о здоровье, и я постараюсь дать полезную информацию на основе медицинских знаний.",
                    "Помните, что я не заменяю врача - при серьезных симптомах обязательно обратитесь за профессиональной помощью."
                ],
                "предупреждения": [
                    "Эта система предоставляет только общую информацию.",
                    "Для точной диагностики и лечения обратитесь к специалисту."
                ]
            },
            "температура": {
                "рекомендации": [
                    "• При температуре 38°C и выше можно принять парацетамол (500-1000 мг для взрослого, не чаще 4 раз в сутки)",
                    "• Обеспечьте обильное питье (2-3 литра теплой жидкости в день: вода, компот, травяной чай)",
                    "• Отдыхайте в прохладном, проветриваемом помещении, используйте легкую одежду",
                    "• Можно использовать физические методы охлаждения: прохладные компрессы на лоб, обтирания водой комнатной температуры"
                ],
                "предупреждения": [
                    "• Срочно к врачу при температуре выше 39°C, особенно у детей и пожилых",
                    "• Немедленно обратитесь за помощью при судорогах, потере сознания, затрудненном дыхании",
                    "• К врачу при температуре, держащейся более 3 дней или сопровождающейся сильной головной болью, рвотой"
                ]
            },
            "мазь": {
                "рекомендации": [
                    "• При ушибах и травмах: нанесите мазь с противовоспалительным эффектом (Диклофенак, Ибупрофен гель)",
                    "• Для рассасывания синяков: используйте Гепариновую мазь или Троксерутин",
                    "• Наносите тонким слоем 2-3 раза в день, после нанесения обеспечьте покой поврежденной области",
                    "• Дополнительно: холодный компресс в первые 24 часа, возвышенное положение конечности"
                ],
                "предупреждения": [
                    "• Не наносите на поврежденную кожу, открытые раны",
                    "• Обратитесь к врачу при сильной боли, отеке, онемении, ограничении подвижности",
                    "• К врачу при подозрении на перелом или вывих"
                ]
            },
            "таблетка": {
                "рекомендации": [
                    "• Для обезболивания: Парацетамол (безопасный для большинства, 500-1000 мг до 4 раз в сутки)",
                    "• При воспалении: Ибупрофен (200-400 мг до 3 раз в сутки, принимать после еды)",
                    "• При сильной боли: Диклофенак (50 мг до 2 раз в сутки, строго по инструкции)",
                    "• Всегда принимайте с достаточным количеством воды, не превышайте дозировку"
                ],
                "предупреждения": [
                    "• Не принимайте при аллергии на компоненты, проблемах с желудком, печенью, почками",
                    "• Не комбинируйте разные обезболивающие без назначения врача",
                    "• При беременности, кормлении грудью - только по назначению врача"
                ]
            },
            "боль": {
                "рекомендации": [
                    "• Определите характер боли (острая, ноющая, пульсирующая) и локализацию",
                    "• Для облегчения: местные анестетики (мази, гели), покой, возвышенное положение",
                    "• При умеренной боли: парацетамол или ибупрофен по инструкции",
                    "• Дополнительно: теплые/холодные компрессы в зависимости от типа боли"
                ],
                "предупреждения": [
                    "• Немедленно к врачу при острой, нестерпимой боли",
                    "• Срочно при боли в груди, животе, сопровождающейся тошнотой, рвотой, потоотделением",
                    "• К врачу при боли, не проходящей после приема обезболивающих"
                ]
            },
            "рана": {
                "рекомендации": [
                    "• Промойте рану чистой водой или антисептиком (перекись водорода, хлоргексидин)",
                    "• Нанесите антисептическую мазь (Йоддицерин, Левомеколь) или используйте бактерицидный пластырь",
                    "• Наложите стерильную повязку, меняйте ежедневно",
                    "• При небольшом кровотечении: прижмите чистую ткань на 10-15 минут"
                ],
                "предупреждения": [
                    "• Срочно к врачу при сильном кровотечении, которое не останавливается",
                    "• Немедленно при признаках инфекции: сильный отек, покраснение, гной, повышение температуры",
                    "• К врачу при глубоких ранах, ранах на лице, укусах животных, загрязненных ранах"
                ]
            }
        }

        return templates.get(question_type, {
            "рекомендации": ["Обратитесь к врачу для точной диагностики и индивидуальных рекомендаций"],
            "предупреждения": ["Не занимайтесь самолечением при серьезных симптомах"]
        })

    def get_fallback_response(self, question: str) -> str:
        """Резервный ответ при ошибке генерации"""
        return f"""### Медицинская консультация

Вопрос: {question}

Рекомендации:
Обратитесь к врачу для точной диагностики и индивидуальных рекомендаций.

Точный ответ из базы знаний: Обратитесь к врачу для точной диагностики

Информация из базы знаний: Рекомендуется профессиональная медицинская помощь для оценки состояния.

Предупреждения и когда обращаться к врачу:
• При любых признаках осложнений немедленно обратитесь за помощью
• Не занимайтесь самолечением

Общие рекомендации:
• При устойчивых симптомах обратитесь к врачу
• Соблюдайте режим и пейте больше жидкости
• Не занимайтесь самолечением серьезными препаратами

Важно:
Эта информация носит ознакомительный характер и не заменяет консультацию врача.
При серьезных симптомах немедленно обратитесь за медицинской помощью."""
    
    def get_medical_templates(self, question_type: str) -> dict:
        """Получение шаблонов для медицинских ответов"""
        templates = {
            "мазь": {
                "рекомендации": [
                    "При ушибах можно использовать мази с противовоспалительным эффектом:",
                    "• Диклофенак (Voltaren) - при умеренной боли и воспалении",
                    "• Ибuprofen гель - для местного противовоспалительного эффекта", 
                    "• Гепариновая мазь - для рассасывания синяков",
                    "• Троксерутин (Troxerutin) - для уменьшения отека и синяков",
                    "Дополнительные рекомендации:",
                    "• Холодные компрессы в первые 24 часа после ушиба",
                    "• Покой и возвышенное положение поврежденной конечности",
                    "• При сильной боли можно принять обезболивающее внутрь"
                ],
                "предупреждения": [
                    "Обратитесь к врачу если:",
                    "• Сильная боль не проходит после 2-3 дней",
                    "• Появился сильный отек или онемение",
                    "• Нарушена подвижность конечности",
                    "• Есть подозрение на перелом"
                ]
            },
            "таблетка": {
                "рекомендации": [
                    "Для обезболивания при ушибах можно использовать:",
                    "• Парацетамол - безопасный для большинства людей",
                    "• Ибuprofen - также имеет противовоспалительный эффект",
                    "• Диклофенак - более сильное обезболивающее",
                    "Правила приема:",
                    "• Строго по инструкции",
                    "• Не превышать рекомендованную дозировку",
                    "• Принимать после еды для защиты желудка"
                ],
                "предупреждения": [
                    "Не принимайте если:",
                    "• Аллергия на компоненты препарата",
                    "• Проблемы с желудком (язва, гастрит)",
                    "• Беременность (без консультации врача)"
                ]
            },
            "температура": {
                "рекомендации": [
                    "При температуре выше 38.5°C:",
                    "• Парацетамол (Panadol, Efferalgan)",
                    "• Ибuprofen (Nurofen)",
                    "Дополнительные меры:",
                    "• Обильное питье (вода, компот, морс)",
                    "• Покой и проветривание помещения",
                    "• Легкая одежда"
                ],
                "предупреждения": [
                    "Срочно к врачу при:",
                    "• Температуре выше 39°C",
                    "• Судорогах",
                    "• Потере сознания",
                    "• Затрудненном дыхании"
                ]
            },
            "боль": {
                "рекомендации": [
                    "Для облегчения боли:",
                    "• Определите характер и локализацию боли",
                    "• Местные анестетики (мази, гели)",
                    "• Обезболивающие таблетки (по инструкции)",
                    "• Холод/тепло (по показаниям)"
                ],
                "предупреждения": [
                    "Немедленно к врачу при:",
                    "• Острой, нестерпимой боли",
                    "• Боли в груди или животе",
                    "• Боли сопровождающейся тошнотой/рвотой"
                ]
            }
        }
        
        return templates.get(question_type, {
            "рекомендации": ["Обратитесь к врачу для точной диагностики"],
            "предупреждения": ["Не занимайтесь самолечением"]
        })
    
    def generate_contextual_response(self, question: str, context: str, entities: str) -> str:
        """Генерация контекстуального ответа"""
        try:
            response_parts = []
            
            # Определяем тип вопроса
            question_type = self.get_question_type(question)
            
            # Заголовок
            response_parts.append("### Медицинская консультация")
            
            # Вопрос
            response_parts.append(f"\n**Вопрос:** {question}")
            
            # Анализ сущностей
            if entities and "Найденные медицинские сущности:" in entities:
                response_parts.append(f"\n**Анализ сущностей:** {entities}")
            
            # Получаем шаблоны для типа вопроса
            templates = self.get_medical_templates(question_type)
            
            # Добавляем рекомендации из шаблонов
            response_parts.append("\n**Рекомендации:**")
            for rec in templates["рекомендации"]:
                response_parts.append(rec)
            
            # Попытка извлечь точный ответ из найденного контекста
            precise_answer = ""
            if context and len(context) > 50:
                try:
                    precise_answer = self.extract_answer(context, question)
                    # Проверяем, что ответ релевантный
                    if precise_answer and len(precise_answer) > 5:
                        # Если это не "Парацетамол" при вопросе о мазях, добавляем
                        if not (question_type == "мазь" and "парацетамол" in precise_answer.lower()):
                            response_parts.append(f"\n**Точный ответ из базы знаний:** {precise_answer}")
                except Exception as e:
                    print(f"Не удалось извлечь точный ответ: {e}")
            
            # Информация из базы знаний (отфильтрованная)
            if context and len(context) > 50:
                # Извлекаем ключевые предложения (первые 3)
                sentences = context.split(". ")[:3]
                key_info = ". ".join(sentences) + "."
                response_parts.append(f"\n**Информация из базы знаний:** {key_info}")
            
            # Добавляем предупреждения из шаблонов
            response_parts.append("\n**Предупреждения и когда обращаться к врачу:**")
            for warn in templates["предупреждения"]:
                response_parts.append(warn)
            
            # Универсальные рекомендации
            response_parts.append("\n**Общие рекомендации:**")
            response_parts.append("• При устойчивых симптомах обратитесь к врачу")
            response_parts.append("• Соблюдайте режим и пейте больше жидкости")
            response_parts.append("• Не занимайтесь самолечением серьезными препаратами")
            
            # Предупреждение
            response_parts.append("\n**Важно:**")
            response_parts.append("Эта информация носит ознакомительный характер и не заменяет консультацию врача.")
            response_parts.append("При серьезных симптомах немедленно обратитесь за медицинской помощью.")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            return f"Произошла ошибка при формировании ответа: {str(e)}"
    
    def process_question(self, question: str) -> str:
        """Основной метод обработки вопроса"""
        print(f"\nОбработка вопроса: '{question}'")

        try:
            # Определяем тип вопроса
            question_type = self.get_question_type(question)

            # Шаг 1: Анализ медицинских сущностей в вопросе
            print("Шаг 1: Анализ сущностей...")
            entities = self.analyze_entities(question)
            print(f"   Найденные сущности: {entities}")

            # Для приветствий пропускаем поиск в базе знаний
            if question_type == "приветствие":
                knowledge = ""
                print("   Пропуск поиска в базе знаний для приветствия")
            else:
                # Шаг 2: Поиск релевантной информации в базе знаний
                print("Шаг 2: Поиск в базе знаний...")
                knowledge = self.search_knowledge(question)
                print(f"   Найдено информации: {len(knowledge)} символов")

            # Шаг 3: Генерация ответа с помощью генеративной модели
            print("Шаг 3: Генерация ответа...")
            final_response = self.generate_generative_response(question, knowledge, entities)

            print("OK Обработка завершена")
            return final_response

        except Exception as e:
            print(f"ERROR Ошибка при обработке вопроса: {e}")
            return f"""
Произошла ошибка при обработке вашего вопроса.

**Что вы можете сделать:**
1. Переформулируйте вопрос более конкретно
2. Укажите конкретные симптомы или препараты
3. Обратитесь к врачу при серьезных симптомах

**Ошибка:** {str(e)}
"""

# Тестирование
if __name__ == "__main__":
    print("Тестирование продвинутого медицинского ассистента")
    print("=" * 60)
    
    # Создаем экземпляр ассистента
    assistant = AdvancedMedicalAssistant()
    
    if assistant.models_loaded:
        # Тестовые вопросы
        test_questions = [
            "палец обрезал что делать?",
            "Ногу ушиб какая мазь нужна?",
            "Что делать при температуре 38.5?",
            "Можно ли принимать парацетамол при головной боли?",
            "Какие симптомы при гипертонии?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\nТест {i}: {question}")
            print("-" * 50)
            
            response = assistant.process_question(question)
            print(f"Ответ:\n{response}")
            print("\n" + "=" * 60)
    else:
        print("Не удалось загрузить модели. Проверьте установку зависимостей.")