// SPDX-License-Identifier: MIT

pragma solidity 0.8.23;

import { Ownable } from "../lib/openzeppelin-contracts/contracts/access/Ownable.sol";
import { Create3 } from "./libraries/Create3.sol";

contract Create3Deployer is Ownable {
    constructor() Ownable(msg.sender) {} // solhint-disable-line no-empty-blocks

    function deploy(bytes32 salt, bytes calldata code) external onlyOwner returns (address) {
        return Create3.create3(salt, code);
    }

    function addressOf(bytes32 salt) external view returns (address) {
        return Create3.addressOf(salt);
    }
}