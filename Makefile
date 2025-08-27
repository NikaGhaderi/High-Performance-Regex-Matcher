# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=gnu11
LDFLAGS = -lhs

# Hyperscan paths
HS_INCLUDE = /usr/include/hs
HS_LIB = /usr/lib/x86_64-linux-gnu

# Add Hyperscan include path
CFLAGS += -I$(HS_INCLUDE)
LDFLAGS += -L$(HS_LIB)

# Project name
TARGET = simple_hs_matcher

# Source files
SRCS = simple_hs_matcher.c
OBJS = $(SRCS:.c=.o)

# Default target
all: $(TARGET)

# Create the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Install dependencies
install-deps:
	sudo apt update
	sudo apt install -y libhyperscan-dev

# Clean build files
clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean install-deps