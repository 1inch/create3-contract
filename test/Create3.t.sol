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
        bytes memory proxyBytecode = hex"68363d3d37363d34f0ff3d5260096017f3";
        assertEq(
            keccak256(proxyBytecode),
            0x8d04f296f449a1e795ad35f27e6b1d09af5a2422fa137f3d6cbf52d7a920975c
        );
    }

    function test_owner() external view {
        assertEq(deployer.owner(), address(this));
    }

    /// Cross-checked with the Rust miner unit test (create3_address_known_vector):
    /// factory 0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf + zero salt
    /// => 0x7f35ba6cce28fdd976c66589f2e109a6fb69ad27
    function test_addressOf_zeroSalt() external {
        Create3Deployer fixedDeployer = new Create3Deployer();
        vm.etch(0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf, address(fixedDeployer).code);
        Create3Deployer fixed_ = Create3Deployer(0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf);
        assertEq(
            fixed_.addressOf(bytes32(0)),
            0x7F35BA6cCe28FdD976c66589F2E109A6fB69aD27
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
