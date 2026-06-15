// SPDX-License-Identifier: MIT

pragma solidity 0.8.23;

import { Script, console } from "../lib/forge-std/src/Script.sol";
import { Create3Deployer } from "../contracts/Create3Deployer.sol";

contract DeployCreate3Deployer is Script {
    function run() external {
        vm.startBroadcast();
        Create3Deployer deployer = new Create3Deployer();
        vm.stopBroadcast();

        console.log("Create3Deployer deployed at:", address(deployer));
        console.log("Use this address as the <factory> argument for the miner.");
    }
}
