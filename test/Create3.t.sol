// SPDX-License-Identifier: MIT

pragma solidity 0.8.23;

import { Test } from "../lib/forge-std/src/Test.sol";
import { Ownable } from "../lib/openzeppelin-contracts/contracts/access/Ownable.sol";
import { Create3Deployer } from "../contracts/Create3Deployer.sol";

contract Dummy {}

contract Create3Test is Test {
    Create3Deployer internal deployer;

    function setUp() external {
        deployer = new Create3Deployer();
    }

    /// Guards the hash comment in contracts/libraries/Create3.sol
    function test_proxyBytecodeHash() external pure {
        bytes memory proxyBytecode = hex"67363d3d37363d34f03d5260086018f3";
        assertEq(
            keccak256(proxyBytecode),
            0x21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f
        );
    }

    function test_owner() external view {
        assertEq(deployer.owner(), address(this));
    }

    /// Cross-checked with the Rust miner unit test (create3_address_known_vector):
    /// factory 0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf + zero salt
    /// => 0x6c8ed9dc3734d7944beddd2fb5acdf5f17247870
    function test_addressOf_zeroSalt() external {
        Create3Deployer fixedDeployer = new Create3Deployer();
        vm.etch(0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf, address(fixedDeployer).code);
        Create3Deployer fixed_ = Create3Deployer(0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf);
        assertEq(
            fixed_.addressOf(bytes32(0)),
            0x6c8Ed9dC3734d7944BEDDd2fB5AcdF5f17247870
        );
    }

    function test_deploy_landsAtAddressOf() external {
        bytes32 salt = keccak256("create3-miner test salt");
        address predicted = deployer.addressOf(salt);

        address deployed = deployer.deploy(salt, type(Dummy).creationCode);

        assertEq(deployed, predicted);
        assertGt(deployed.code.length, 0);
    }

    function test_deploy_revertsForNonOwner() external {
        vm.prank(address(0xBEEF));
        vm.expectRevert(
            abi.encodeWithSelector(Ownable.OwnableUnauthorizedAccount.selector, address(0xBEEF))
        );
        deployer.deploy(bytes32(0), type(Dummy).creationCode);
    }
}
