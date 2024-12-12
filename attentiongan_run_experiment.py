import torch
import torch.nn as nn
import torchvision
import torch.optim
import argparse
import image_data_loader
import attentiongandehazeNet


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(args):
    # 初始化生成器和判别器
    ld_net = attentiongandehazeNet.attentionDehaze_Net().cuda()
    ld_net.apply(weights_init)
    dis = attentiongandehazeNet.Discriminator().cuda()
    dis.apply(weights_init)

    # 数据加载
    training_data = image_data_loader.hazy_data_loader(args["train_original"], args["train_hazy"])
    validation_data = image_data_loader.hazy_data_loader(args["train_original"], args["train_hazy"], mode="val")
    training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # 损失函数
    criterion_mse = nn.MSELoss().cuda()  # 像素损失
    criterion_gan = nn.BCELoss().cuda()  # 对抗损失

    # 优化器
    gen_optimizer = torch.optim.Adam(ld_net.parameters(), lr=float(args["learning_rate"]), weight_decay=0.0001)
    dis_optimizer = torch.optim.Adam(dis.parameters(), lr=float(args["learning_rate"]), weight_decay=0.0001)

    # 训练模式
    ld_net.train()
    dis.train()

    # 训练超参数
    num_of_epochs = int(args["epochs"])
    lambda_adv = 0.5  # 对抗损失权重
    lambda_mse = 2.5  # MSE 损失权重

    best_val_loss = float("inf")  # 用于记录最佳验证损失
    best_gen_state_dict = None  # 记录最佳生成器权重
    best_dis_state_dict = None  # 记录最佳判别器权重

    for epoch in range(num_of_epochs):
        for iteration, (hazefree_image, hazy_image) in enumerate(training_data_loader):
            hazefree_image = hazefree_image.cuda()
            hazy_image = hazy_image.cuda()

            ### 1. 更新判别器 ###
            dis_optimizer.zero_grad()
            dehaze_image = ld_net(hazy_image)

            # 判别真实图像
            real_output = dis(hazefree_image)
            real_labels = torch.ones_like(real_output)  # 确保与 real_output 形状一致
            real_loss = criterion_gan(real_output, real_labels)

            # 判别生成图像
            fake_output = dis(dehaze_image.detach())
            fake_labels = torch.zeros_like(fake_output)  # 确保与 fake_output 形状一致
            fake_loss = criterion_gan(fake_output, fake_labels)

            # 判别器总损失
            dis_loss = (real_loss + fake_loss) / 2
            dis_loss.backward()
            dis_optimizer.step()

            ### 2. 更新生成器 ###
            gen_optimizer.zero_grad()

            # 对抗损失
            fake_output = dis(dehaze_image)
            adv_loss = criterion_gan(fake_output, real_labels)

            # 像素损失
            mse_loss = criterion_mse(dehaze_image, hazefree_image)

            # 生成器总损失
            gen_loss = lambda_adv * adv_loss + lambda_mse * mse_loss
            gen_loss.backward()
            gen_optimizer.step()

            # 日志输出
            if ((iteration + 1) % 20) == 0:
                print(f"Epoch [{epoch+1}/{num_of_epochs}], Step [{iteration+1}/{len(training_data_loader)}], "
                      f"D Loss: {dis_loss.item():.4f}, G Loss: {gen_loss.item():.4f}")

        ### 验证阶段 ###
        ld_net.eval()
        total_val_loss = 0
        with torch.no_grad():
            for hazefree_image, hazy_image in validation_data_loader:
                hazefree_image = hazefree_image.cuda()
                hazy_image = hazy_image.cuda()

                dehaze_image = ld_net(hazy_image)
                val_loss = criterion_mse(dehaze_image, hazefree_image)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(validation_data_loader)
        print(f"Epoch [{epoch+1}/{num_of_epochs}], Validation Loss: {avg_val_loss:.4f}")

        # 保存最佳模型权重
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_gen_state_dict = ld_net.state_dict()
            best_dis_state_dict = dis.state_dict()
            print(f"New best model found! Saving...")
            torch.save(best_gen_state_dict, "trained_attentiongan_weights/best_ld_net.pth")
            torch.save(best_dis_state_dict, "trained_attentiongan_weights/best_dis.pth")
        ld_net.train()

    # 保存最佳模型权重

    print(f"Best model saved with Validation Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-th", "--train_hazy", default='/mnt/Gancon/data/training_data/', help="path to hazy training images")
    ap.add_argument("-to", "--train_original",  default='/mnt/Gancon/data/orginal_data/', help="path to original training images")
    ap.add_argument("-e", "--epochs", required=True, help="number of epochs for training")
    ap.add_argument("-lr", "--learning_rate", required=True, help="learning rate for training")

    args = vars(ap.parse_args())
    train(args)
