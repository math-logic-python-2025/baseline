# Математическая логика на Python [2025]

## Оглавление

1. [Ссылки](#Ссылки)
2. [Как сдавать ДЗ?](#Как-сдавать-ДЗ?)
3. [Локальное тестирование](#Локальное-тестирование)
4. [ВАЖНО ЗНАТЬ](#важно-знать)

## Ссылки
[HSE Wiki](http://wiki.cs.hse.ru/%D0%9C%D0%B0%D1%82%D0%B5%D0%BC%D0%B0%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B0%D1%8F_%D0%BB%D0%BE%D0%B3%D0%B8%D0%BA%D0%B0_%D0%BD%D0%B0_Python_25) \
[Telegram канал](https://t.me/+IRoFqPGn2ZFhZWMy) \
[Telegram чат](https://t.me/+bxpcEC7MnsExYjUy) \
[Google Classroom](https://classroom.google.com/c/NzI1NTIzNTE3MjA0?cjc=frttno2) [archived]

## Как сдавать ДЗ?

1. Получить ссылку на ДЗ
2. Склонировать репозиторий (для каждого ДЗ новый)
3. Реализовать функционал
4. Протестировать свой код (см. раздел [Локальное тестирование](README.md#локальное-тестирование))
5. Запушить в `main` ветку

## Локальное тестирование

### Environment

Создайте виртуальное окружение:
```bash
python3 -m venv venv
```

Активируйте его (Linux/Mac):
```bash
source venv/bin/activate
```

Активируйте его (Windows):
```bash
venv\Scripts\activate
```

Установите `pytest`:
```bash
pip install pytest
```

### Запускаем тесты

Полное тестирование всех ДЗ:
```bash
pytest tests/hw/
```

Проверка конкретного ДЗ (пример):
```bash
pytest tests/hw/test_hw01.py
```

Проверка конкретного номера в ДЗ (пример):
```bash
pytest tests/hw/test_hw01.py::test_task1
```

## Важно знать
### <font color="red">Запрещено</font> изменять тесты
Если по истечении дедлайна будет изменён какой-либо тест в `tests/*`, ДЗ будет обнулено.

Если вы случайно что-то изменили, возьмите тесты с предыдущей версии вашего репозитория, где тесты не изменены.

### О предыдущих ДЗ
В каждом N+1-ом ДЗ будут реализованы предыдущие N ДЗ. 

Так что даже если вы пропустите какое ДЗ, вы сможете сдавать последующие без необходимости дорешивания всех предыдущих.

### О системе

Локальное тестирование абсолютно идентично тестированию в системе, если не оговорено иное.
Проверяйте свой код локально, запуск тестов на сервере будет запущен только после дедлайна.