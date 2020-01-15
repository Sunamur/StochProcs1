## StochProcs1
Financial Instruments Pricing project.



# Этапы проекта:
- Экспертная оценка ковариационной матрицы для риск-факторов
- Построение симуляционного блока
- Прайсинг финансовых инструментов
-- Процентный своп
-- Валютный форвард
-- Валютный своп
- Оценка Potential Future Exposure по портфелю

# Комментарии Соколовского:
1. Составляющие: симуляционная модель для риск-факторов (учитывающая корреляции, которые вы оцениваете самостоятельно), 3 прайсера для деривативов, сам расчёт квантильный метрики.
2. Часть оценки стоимости деривативов и метрики PFE становится опциональной - кто успел, оставляйте, кто нет - можно не делать. Вам достаточно насимулировать три скоррелированных риск фактора и для каждого оценить 95% ю квантиль двухнедельного приращения фактора.
