# Makefile для управления контейнерами

# Переменные конфигурации
USER ?= $(shell whoami)
IMAGE_NAME = $(USER)_lit_train
LIGHT_CONTAINER_NAME = lit_container_light
HEAVY_CONTAINER_NAME = lit_container_heavy
WANDB_API_KEY = $(shell cat ~/.wandb/token)# Замените на свой API ключ
DEVICES ?= 0# GPU по умолчанию, можно переопределить при вызове

# Пути для монтирования
HOST_DIR = $(shell pwd)
CONTAINER_DIR = /workspace

# Команды для Docker
_DOCKER_SUDO := $(shell docker ps >/dev/null 2>&1 && echo "" || echo "sudo")
DOCKER = $(_DOCKER_SUDO) docker

# Общие опции сети
NETWORK_OPTS = --network=host

# Выбор контейнера по умолчанию
C ?= light


# Определение имени контейнера в зависимости от выбора
ifeq ($(C),light)
    CONTAINER_NAME = $(LIGHT_CONTAINER_NAME)
else
    CONTAINER_NAME = $(HEAVY_CONTAINER_NAME)
endif


# Информация о конфигурации
.PHONY: info
info:
	@echo "Текущая конфигурация:"
	@echo "  Пользователь: $(USER)"
	@echo "  Имя образа: $(IMAGE_NAME)"
	@echo "  Легкий контейнер: $(LIGHT_CONTAINER_NAME)"
	@echo "  Тяжелый контейнер: $(HEAVY_CONTAINER_NAME)"
	@echo "  Текущий выбранный контейнер: $(CONTAINER_NAME)"
	@echo "  DEVICES: $(DEVICES)"
	@echo "  Текущий WANDB_API_KEY: $(shell echo $(WANDB_API_KEY) | cut -c1-4)...$(shell echo $(WANDB_API_KEY) | rev | cut -c1-4 | rev) ($(shell echo -n $(WANDB_API_KEY) | wc -c) символов)"; \


# Правило для сборки образа
.PHONY: build
build:
	$(DOCKER) build -t $(IMAGE_NAME) ./docker
	@echo "Образ $(IMAGE_NAME) собран успешно"


# Правило для запуска "легкого" контейнера
.PHONY: run-light
run-light:
	$(DOCKER) run -d \
		--name $(LIGHT_CONTAINER_NAME) \
		$(NETWORK_OPTS) \
		-v $(HOST_DIR):$(CONTAINER_DIR) \
		--cpus=6 \
		--memory=20g \
		--shm-size=8g \
		-e WANDB_API_KEY=$(WANDB_API_KEY) \
		$(IMAGE_NAME) \
		sleep infinity
	$(DOCKER) exec $(LIGHT_CONTAINER_NAME) wandb login $(WANDB_API_KEY)
	@echo "Легкий контейнер $(LIGHT_CONTAINER_NAME) запущен. Используйте 'make exec CONTAINER=light CMD=\"команда\"'"


# Правило для запуска "тяжелого" контейнера с GPU
.PHONY: run-heavy
run-heavy:
	$(DOCKER) run -d \
		--name $(HEAVY_CONTAINER_NAME) \
		$(NETWORK_OPTS) \
		-v $(HOST_DIR):$(CONTAINER_DIR) \
		--cpus=12 \
		--memory=40g \
		--shm-size=8g \
		--gpus '"device=$(DEVICES)"' \
		-e WANDB_API_KEY=$(WANDB_API_KEY) \
		$(IMAGE_NAME) \
		sleep infinity
	$(DOCKER) exec $(HEAVY_CONTAINER_NAME) wandb login $(WANDB_API_KEY)
	@echo "Тяжелый контейнер $(HEAVY_CONTAINER_NAME) запущен с GPU $(DEVICES). Используйте 'make exec C=heavy CMD=\"команда\"'"


# Правило для интерактивного подключения к работающему контейнеру
.PHONY: connect
connect:
	@if ! $(DOCKER) ps -q -f name=$(CONTAINER_NAME) | grep -q .; then \
		echo "Контейнер $(CONTAINER_NAME) не запущен"; \
		exit 1; \
	fi
	$(DOCKER) exec -it $(CONTAINER_NAME) bash


# Правило для выполнения произвольной команды в выбранном контейнере (не интерактивно)
.PHONY: exec
exec:
	@if [ -z "$(CMD)" ]; then \
		echo "Ошибка: Укажите команду для выполнения, например 'make exec CMD=\"python script.py > output.log\" C=heavy'"; \
		exit 1; \
	fi
	@if ! $(DOCKER) ps -q -f name=$(CONTAINER_NAME) | grep -q .; then \
		echo "Контейнер $(CONTAINER_NAME) не запущен"; \
		exit 1; \
	fi
	$(DOCKER) exec -d $(CONTAINER_NAME) bash -c "$(CMD)"
	@echo "Команда запущена в контейнере $(CONTAINER_NAME)"


# Команда для запуска обучения с логированием
.PHONY: run-train
run-train:
	@if ! $(DOCKER) ps -q -f name=$(CONTAINER_NAME) | grep -q .; then \
		echo "Контейнер $(CONTAINER_NAME) не запущен"; \
		exit 1; \
	fi
	@echo "Запуск обучения в контейнере $(CONTAINER_NAME)..."
	@echo "Логи будут сохранены в файл run_train.log"
	$(DOCKER) exec -d $(CONTAINER_NAME) bash -c "python3.11 run_train.py > run_train.log 2>&1"
	@echo "Команда обучения запущена. Используйте 'make tail-log' для отслеживания логов."


# Команда для отслеживания логов обучения
.PHONY: tail-log
tail-log:
	tail -f run_train.log


# Правило для остановки и удаления контейнера
.PHONY: stop
stop:
	@if ! $(DOCKER) ps -a -q -f name=$(CONTAINER_NAME) | grep -q .; then \
		echo "Контейнер $(CONTAINER_NAME) не существует"; \
		exit 0; \
	fi
	$(DOCKER) stop $(CONTAINER_NAME)
	$(DOCKER) rm $(CONTAINER_NAME)
	@echo "Контейнер $(CONTAINER_NAME) остановлен и удален."


# Правило для быстрой остановки и удаления контейнера (принудительно)
.PHONY: fstop
fstop:
	@if ! $(DOCKER) ps -a -q -f name=$(CONTAINER_NAME) | grep -q .; then \
		echo "Контейнер $(CONTAINER_NAME) не существует"; \
		exit 0; \
	fi
	$(DOCKER) rm -f $(CONTAINER_NAME)
	@echo "Контейнер $(CONTAINER_NAME) принудительно удален."


# Правило для остановки всех контейнеров
.PHONY: stop-all
stop-all:
	-$(DOCKER) stop $(LIGHT_CONTAINER_NAME) 2>/dev/null || true
	-$(DOCKER) rm $(LIGHT_CONTAINER_NAME) 2>/dev/null || true
	-$(DOCKER) stop $(HEAVY_CONTAINER_NAME) 2>/dev/null || true
	-$(DOCKER) rm $(HEAVY_CONTAINER_NAME) 2>/dev/null || true
	@echo "Все контейнеры остановлены и удалены."


# Правило для быстрой остановки всех контейнеров (принудительно)
.PHONY: fstop-all
fstop-all:
	-$(DOCKER) rm -f $(LIGHT_CONTAINER_NAME) $(HEAVY_CONTAINER_NAME) 2>/dev/null || true
	@echo "Все контейнеры принудительно удалены."


# Проверка статуса контейнеров
.PHONY: status
status:
	@echo "Статус контейнеров для образа $(IMAGE_NAME):"
	@echo "-------------------"
	@echo -n "Легкий контейнер ($(LIGHT_CONTAINER_NAME)): "
	@if $(DOCKER) ps -q -f name=$(LIGHT_CONTAINER_NAME) | grep -q .; then \
		echo "ЗАПУЩЕН"; \
	elif $(DOCKER) ps -a -q -f name=$(LIGHT_CONTAINER_NAME) | grep -q .; then \
		echo "ОСТАНОВЛЕН"; \
	else \
		echo "НЕ СУЩЕСТВУЕТ"; \
	fi
	@echo -n "Тяжелый контейнер ($(HEAVY_CONTAINER_NAME)): "
	@if $(DOCKER) ps -q -f name=$(HEAVY_CONTAINER_NAME) | grep -q .; then \
		echo "ЗАПУЩЕН"; \
	elif $(DOCKER) ps -a -q -f name=$(HEAVY_CONTAINER_NAME) | grep -q .; then \
		echo "ОСТАНОВЛЕН"; \
	else \
		echo "НЕ СУЩЕСТВУЕТ"; \
	fi


# Просмотр логов контейнера
.PHONY: logs
logs:
	@if ! $(DOCKER) ps -a -q -f name=$(CONTAINER_NAME) | grep -q .; then \
		echo "Контейнер $(CONTAINER_NAME) не существует"; \
		exit 1; \
	fi
	$(DOCKER) logs $(if $(FOLLOW),-f,) $(CONTAINER_NAME)


# Правило для комплексной очистки (остановка контейнеров + удаление образа)
.PHONY: hard-clean
hard-clean: stop-all
	-$(DOCKER) rmi $(IMAGE_NAME)
	@echo "Образ $(IMAGE_NAME) удален."


# Помощь по командам
.PHONY: help
help:
	@echo "Управление контейнерами для $(USER)_lit_train"
	@echo "=============================================="
	@echo "Доступные команды:"
	@echo "  info                        - Показать текущую конфигурацию"
	@echo "  build                       - Собрать Docker образ $(IMAGE_NAME)"
	@echo "  run-light                   - Запустить легкий контейнер $(LIGHT_CONTAINER_NAME)"
	@echo "  run-heavy [DEVICES=0,1] - Запустить тяжелый контейнер $(HEAVY_CONTAINER_NAME) с указанными GPU"
	@echo "  connect [CONTAINER=light|heavy] - Интерактивное подключение к указанному контейнеру"
	@echo "  exec CMD=\"команда\" [CONTAINER=light|heavy] - Выполнить команду в контейнере (не интерактивно)"
	@echo "  run-train [CONTAINER=light|heavy] - Запустить обучение с логированием в run_train.log"
	@echo "  tail-log [CONTAINER=light|heavy]  - Отслеживать логи обучения в реальном времени"
	@echo "                           - Запустить Python-скрипт в контейнере с перенаправлением вывода"
	@echo "  stop [CONTAINER=light|heavy] - Остановить и удалить указанный контейнер"
	@echo "  fstop [CONTAINER=light|heavy] - Принудительно удалить указанный контейнер (быстрее, но без корректного завершения)"
	@echo "  stop-all                   - Остановить и удалить все контейнеры"
	@echo "  fstop-all                  - Принудительно удалить все контейнеры (быстрее, но без корректного завершения)"
	@echo "  logs [FOLLOW=1] [CONTAINER=light|heavy] - Вывести логи контейнера (с FOLLOW=1 режим слежения)"
	@echo "  status                     - Проверить статус контейнеров"
	@echo "  hard-clean                      - Удалить все контейнеры и образ"
	@echo ""
	@echo "Примеры:"
	@echo "  make run-heavy DEVICES=0,1"
	@echo "  make exec CMD=\"nvidia-smi\" CONTAINER=heavy"
	@echo "  make logs FOLLOW=1 CONTAINER=heavy"
	@echo "  make fstop-all              - Быстрое принудительное удаление всех контейнеров"
