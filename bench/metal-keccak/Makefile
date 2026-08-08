BIN := keccak-bench

$(BIN): main.swift
	swiftc -O -o $@ main.swift

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN)

.PHONY: run clean
